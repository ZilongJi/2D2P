function R = startRotaryLogger(varargin)
% startRotaryLogger (Standalone, batch write) - single-file version
% Stop by calling:  R.stop();
%
% REQUIREMENT:
%   You MUST pass outDir explicitly:
%       R = startRotaryLogger('outDir', 'D:\path\to\session_folder');
%
% New behavior:
%   - Auto-stop previous logger if you forgot to call R.stop()
%   - Uses a tagged timer so only our own timer is auto-cleaned
%
% Output columns (fixed-width):
%   1) DateTime        (derived wall-clock: DT0 + MonotonicSec)
%   2) MonotonicSec    (toc(GLOBAL_MONO_TIC))
%   3) AngleDeg        (mod(count/countsPerRev*360, 360))

    % ===== parse required outDir (MUST be provided) =====
    if nargin < 2
        error('startRotaryLogger:MissingOutDir', ...
              'You must call startRotaryLogger(''outDir'', <folder>).');
    end
    if ~(ischar(varargin{1}) || isstring(varargin{1})) || ~strcmpi(char(varargin{1}), 'outDir')
        error('startRotaryLogger:InvalidSyntax', ...
              'Expected first argument ''outDir''. Example: startRotaryLogger(''outDir'',''D:\\data\\session1'')');
    end
    outDir = varargin{2};
    if ~(ischar(outDir) || isstring(outDir))
        error('startRotaryLogger:InvalidOutDir', 'outDir must be a char or string.');
    end
    outDir = char(outDir);
    if ~exist(outDir,'dir')
        mkdir(outDir);
    end
    % ===================================================

    % ======= AUTO CLEANUP PREVIOUS INSTANCE =======
    persistent prevStopFcn
    if ~isempty(prevStopFcn)
        try
            prevStopFcn();   % flush + close file + close serial + delete timer
        catch
        end
        prevStopFcn = [];
    end

    % Also clean any leftover timers from earlier crashes (ONLY ours)
    try
        tt = timerfindall('Tag','RotaryLoggerTimer');
        if ~isempty(tt)
            try, stop(tt); catch, end
            try, delete(tt); catch, end
        end
    catch
    end
    % =============================================

    % ========= user config =========
    cfg.comPort      = 'COM7';
    cfg.countsPerRev = 36800;
    cfg.sampleHz     = 200;

    cfg.outDir       = outDir;             % <-- REQUIRED from caller
    cfg.baseName     = 'rotary_stream';

    cfg.serialTimeoutSec = 0.005;          % try 0.005 / 0.01
    cfg.batchN       = 500;

    % DateTime formatting (datestr)
    cfg.dateTimeFormat = 'yyyy-mm-dd HH:MM:SS.FFF';
    % ===============================

    % global monotonic (do NOT reset)
    global GLOBAL_MONO_TIC
    if isempty(GLOBAL_MONO_TIC)
        GLOBAL_MONO_TIC = tic;
        fprintf('[RotaryLogger] GLOBAL_MONO_TIC was empty; initialized now.\n');
    end

    % Wall-clock epoch captured once; DateTime column is derived:
    %   DateTime = DT0 + seconds(MonotonicSec)
    % Avoids per-tick OS time reads (less jitter, no NTP jumps).
    DT0 = datetime('now');

    % close leftovers (best-effort) — OK to keep, but note it closes all serial objects
    try
        objs = instrfind;
        if ~isempty(objs)
            try, fclose(objs); catch, end
            try, delete(objs); catch, end
        end
    catch
    end

    % open COM
    maxAttempts = 5;
    E2019Q_ID = [];
    for k = 1:maxAttempts
        try
            E2019Q_ID = E2019Q.Open_COM_Port(cfg.comPort);
            break;
        catch
            pause(0.5);
            if k == maxAttempts
                error('Failed to open %s after %d attempts.', cfg.comPort, maxAttempts);
            end
        end
    end

    % serial settings
    try
        E2019Q_ID.Timeout = cfg.serialTimeoutSec;
        flushinput(E2019Q_ID);
    catch ME
        fprintf(2, '[RotaryLogger] Failed to set serial Timeout/flushinput: %s\n', ME.message);
    end

    % create logfile
    ts = datestr(now,'yyyymmdd_HHMMSS');
    outTxt = fullfile(cfg.outDir, sprintf('%s_%s.txt', cfg.baseName, ts));
    fid = fopen(outTxt,'w');
    if fid == -1
        try, E2019Q.Close_COM_Port(E2019Q_ID); catch, end
        try, delete(E2019Q_ID); catch, end
        error('Cannot open output file: %s', outTxt);
    end

    % Header (fixed-width aligned)
    fprintf(fid, '%-23s  %12s  %10s\n', 'DateTime', 'MonotonicSec', 'AngleDeg');

    % ---- LOCALS for speed (no R.* in hot path) ----
    batchN = max(1, round(cfg.batchN));
    tBuf  = nan(batchN, 1);
    aBuf  = nan(batchN, 1);
    dtBuf = strings(batchN, 1);
    bIdx  = 0;

    % stop state flags
    isStopped = false;

    % build public state
    R = struct();
    R.cfg       = cfg;
    R.E2019Q_ID = E2019Q_ID;
    R.fid       = fid;
    R.timer     = [];
    R.outTxt    = outTxt;
    R.stop      = @stopThis;     % <-- stop handle you call as R.stop()

    % publish stop handle for next invocation auto-cleanup
    prevStopFcn = @stopThis;

    % timer
    R.timer = timer( ...
        'ExecutionMode','fixedRate', ...
        'Period', 1/cfg.sampleHz, ...
        'BusyMode','drop', ...
        'TimerFcn',@(~,~) onTick(), ...
        'ErrorFcn',@(~,e) onTimerError(e));

    % tag this timer so we can clean only ours
    try
        R.timer.Tag = 'RotaryLoggerTimer';
    catch
    end

    start(R.timer);

    fprintf('[RotaryLogger] Started.\n');
    fprintf('[RotaryLogger] Writing: %s\n', outTxt);
    fprintf('[RotaryLogger] sampleHz=%.1f | Timeout=%.3f s | batchN=%d\n', ...
        cfg.sampleHz, cfg.serialTimeoutSec, batchN);

    % ================= nested =================

    function onTick()
        if isStopped || fid == -1
            return
        end
    
        % MATLAB-process monotonic time (global, do NOT reset)
        tMono = toc(GLOBAL_MONO_TIC);
    
        % Real system wall-clock time (read OS clock every tick)
        dtNow = datetime('now');
        dtStr = datestr(dtNow, cfg.dateTimeFormat);
    
        ang   = readAngleDeg_FAST(E2019Q_ID, cfg.countsPerRev);
    
        bIdx = bIdx + 1;
        if bIdx > batchN
            bIdx = batchN;
        end
    
        dtBuf(bIdx) = dtStr;
        tBuf(bIdx)  = tMono;
        aBuf(bIdx)  = ang;
    
        if bIdx == batchN
            flushBatch();
        end
    end

    function flushBatch()
        nWrite = bIdx;
        if nWrite <= 0 || fid == -1 || isStopped
            bIdx = 0;
            return
        end

        try
            for k = 1:nWrite
                fprintf(fid, '%-23s  %12.6f  %10.6f\n', dtBuf(k), tBuf(k), aBuf(k));
            end
        catch
            % stop on write failure
            try, stopThis(); catch, end
            return
        end

        bIdx = 0;
    end

    function stopThis()
        % safe to call multiple times
        if isStopped
            return
        end
        isStopped = true;

        % detach persistent so next start won't re-call an already-stopped handle
        try
            if isequal(prevStopFcn, @stopThis)
                prevStopFcn = [];
            end
        catch
        end

        % stop timer first
        try
            if ~isempty(R.timer) && isvalid(R.timer)
                stop(R.timer);
                delete(R.timer);
            end
        catch
        end

        % flush remaining
        try
            if fid ~= -1 && bIdx > 0
                for k = 1:bIdx
                    fprintf(fid, '%-23s  %12.6f  %10.6f\n', dtBuf(k), tBuf(k), aBuf(k));
                end
                bIdx = 0;
            end
        catch
        end

        % close file
        try
            if fid ~= -1
                fclose(fid);
            end
            fid = -1;
        catch
        end

        % close serial
        try
            if ~isempty(E2019Q_ID)
                try, E2019Q.Close_COM_Port(E2019Q_ID); catch, end
                if isvalid(E2019Q_ID), delete(E2019Q_ID); end
            end
        catch
        end

        fprintf('[RotaryLogger] Stopped. File saved: %s\n', outTxt);
    end

    function onTimerError(e)
        try
            fprintf(2,'[RotaryLogger] TIMER ERROR: %s\n', e.Data.message);
        catch
        end
        try
            stopThis();
        catch
        end
    end
end

function ang = readAngleDeg_FAST(E2019Q_ID, countsPerRev)
    ang = NaN;
    try
        c = E2019Q.GetEncCountFAST(E2019Q_ID);
        ang = mod(double(c)/countsPerRev*360.0, 360.0);
    catch
        ang = NaN;
    end
end
