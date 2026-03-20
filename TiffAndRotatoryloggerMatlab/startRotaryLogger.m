function R = startRotaryLogger(varargin)
% Minimal rotary logger for running alongside ScanImage.
% Usage:
%   R = startRotaryLogger('outDir', 'D:\data\session1');
%   R = startRotaryLogger('outDir', 'D:\data\session1', 'comPort', 'COM7');
%   R = startRotaryLogger('outDir', 'D:\data\session1', 'comPort', 'COM7', 'sampleHz', 50, 'batchN', 300);
%   ...
%   R.stop();
%
% Output columns:
%   DateTime, MonotonicSec, AngleDeg

    if nargin < 2
        error('startRotaryLogger:MissingOutDir', ...
            'Use: startRotaryLogger(''outDir'', <folder>).');
    end
    if ~(ischar(varargin{1}) || isstring(varargin{1})) || ~strcmpi(char(varargin{1}), 'outDir')
        error('startRotaryLogger:InvalidSyntax', ...
            'First argument must be ''outDir''.');
    end

    outDir = char(varargin{2});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % Minimal config
    cfg.comPort = 'COM7';
    cfg.countsPerRev = 36800;
    cfg.sampleHz = 100;
    cfg.serialTimeoutSec = 0.02;
    cfg.baseName = 'rotary_stream';
    cfg.outDir = outDir;
    cfg.batchN = 100;

    % Optional name-value pairs:
    %   'comPort', 'COM7'
    %   'sampleHz', 50
    %   'countsPerRev', 36800
    %   'batchN', 300
    if nargin > 2
        if mod(nargin - 2, 2) ~= 0
            error('startRotaryLogger:InvalidSyntax', ...
                'Optional arguments must be name-value pairs.');
        end
        for i = 3:2:nargin
            key = lower(char(varargin{i}));
            val = varargin{i + 1};
            switch key
                case 'comport'
                    cfg.comPort = char(val);
                case 'samplehz'
                    cfg.sampleHz = double(val);
                case 'countsperrev'
                    cfg.countsPerRev = double(val);
                case 'batchn'
                    cfg.batchN = max(1, round(double(val)));
                otherwise
                    error('startRotaryLogger:UnknownOption', ...
                        'Unknown option: %s', char(varargin{i}));
            end
        end
    end

    % Clean only our own previous timer instance
    try
        oldTimers = timerfindall('Tag', 'RotaryLoggerTimerSimple');
        if ~isempty(oldTimers)
            stop(oldTimers);
            delete(oldTimers);
        end
    catch
    end

    % Clean stale serial objects on this COM from previous runs
    try
        stale = instrfind('Port', cfg.comPort);
        if ~isempty(stale)
            try, fclose(stale); catch, end
            try, delete(stale); catch, end
        end
    catch
    end

    % Open COM
    sid = E2019Q.Open_COM_Port(cfg.comPort);
    try
        sid.Timeout = cfg.serialTimeoutSec;
        flushinput(sid);
    catch
    end

    % Open output file
    ts = datestr(now, 'yyyymmdd_HHMMSS');
    outTxt = fullfile(cfg.outDir, sprintf('%s_%s.txt', cfg.baseName, ts));
    fid = fopen(outTxt, 'w');
    if fid == -1
        try, E2019Q.Close_COM_Port(sid); catch, end
        try, delete(sid); catch, end
        error('startRotaryLogger:OpenFileFailed', 'Cannot open %s', outTxt);
    end
    fprintf(fid, '%-23s  %12s  %10s\n', 'DateTime', 'MonotonicSec', 'AngleDeg');

    % Use the same global monotonic clock as OnlineGrab for alignment.
    global GLOBAL_MONO_TIC
    if isempty(GLOBAL_MONO_TIC)
        GLOBAL_MONO_TIC = tic;
    end
    monoStartSec = toc(GLOBAL_MONO_TIC);
    wallStartDatenum = now;
    isStopped = false;
    batchN = cfg.batchN;
    dtBuf = strings(batchN, 1);
    tBuf = nan(batchN, 1);
    aBuf = nan(batchN, 1);
    bIdx = 0;

    % Public handle
    R = struct();
    R.cfg = cfg;
    R.outTxt = outTxt;
    R.stop = @stopThis;
    R.timer = [];

    % Timer
    R.timer = timer( ...
        'ExecutionMode', 'fixedRate', ...
        'Period', 1 / cfg.sampleHz, ...
        'BusyMode', 'drop', ...
        'TimerFcn', @(~, ~) onTick(), ...
        'ErrorFcn', @(~, e) onTimerError(e));
    R.timer.Tag = 'RotaryLoggerTimerSimple';
    start(R.timer);

    fprintf('[RotaryLoggerSimple] Started: %s\n', outTxt);
    fprintf('[RotaryLoggerSimple] com=%s sampleHz=%.1f batchN=%d\n', cfg.comPort, cfg.sampleHz, batchN);

    function onTick()
        if isStopped || fid == -1
            return
        end

        tMono = toc(GLOBAL_MONO_TIC);
        dtStr = datestr(wallStartDatenum + (tMono - monoStartSec) / 86400, 'yyyy-mm-dd HH:MM:SS.FFF');
        ang = readAngleDeg_FAST(sid, cfg.countsPerRev);

        bIdx = bIdx + 1;
        if bIdx > batchN
            bIdx = batchN;
        end
        dtBuf(bIdx) = dtStr;
        tBuf(bIdx) = tMono;
        aBuf(bIdx) = ang;

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
            bIdx = 0;
        catch
            stopThis();
        end
    end

    function onTimerError(e)
        try
            fprintf(2, '[RotaryLoggerSimple] TIMER ERROR: %s\n', e.Data.message);
        catch
        end
        stopThis();
    end

    function stopThis()
        if isStopped
            return
        end
        isStopped = true;

        try
            if ~isempty(R.timer) && isvalid(R.timer)
                stop(R.timer);
                delete(R.timer);
            end
        catch
        end

        try
            flushBatch();
        catch
        end

        try
            if fid ~= -1
                fclose(fid);
            end
            fid = -1;
        catch
        end

        try
            if ~isempty(sid)
                try, E2019Q.Close_COM_Port(sid); catch, end
                try
                    if isvalid(sid), delete(sid); end
                catch
                end
            end
        catch
        end

        fprintf('[RotaryLoggerSimple] Stopped.\n');
    end
end

function ang = readAngleDeg_FAST(sid, countsPerRev)
    ang = NaN;
    try
        c = E2019Q.GetEncCountFAST(sid);
        if isempty(c) || ~isscalar(c) || ~isfinite(c)
            return
        end
        ang = mod(double(c) / countsPerRev * 360.0, 360.0);
    catch
        ang = NaN;
    end
end
