classdef RollingLastFramesSaver < handle
    % RollingLastFramesSaver (TIFF-only + global monotonic + epoch-safe buffer)
    %
    % 输出 TXT 三列（不改变定义）：
    %   1) ScanImageTimeSec : raw ScanImage frameTimestamp（秒）
    %   2) MonotonicSec     : global monotonic（toc(GLOBAL_MONO_TIC)，不重置）
    %   3) FrameIndex       : 每次 Focus/Grab 后从 0 开始
    %
    % 新增行为：
    %   - 检测到 Focus/Grab（raw frameTimestamp 回跳/归零）时：
    %       * 清空 ring buffer 状态，避免跨 epoch 混帧输出
    %       * FrameIndex 重置为 0
    %   - 不重置 GLOBAL_MONO_TIC（保证与后续 rotary 线程共享同一时间基准）

    properties
        hSI
        listener_frameAcquired

        % ====== 可配置参数 ======
        N (1,1) double {mustBePositive, mustBeInteger} = 500
        channelToUse (1,1) double {mustBePositive, mustBeInteger} = 2   % CH2
        saveEverySec (1,1) double {mustBePositive} = 120                % 每 120 秒保存一次

        outDir (1,:) char = 'D:\data\ImagingData\TIFF Export'
        baseName (1,:) char = 'online_grab'

        onlySaveWhenBufferFilled (1,1) logical = true

        % ====== Focus/Grab 检测阈值（秒）=====
        % raw frameTimestamp 比上一帧小超过该阈值 => 判定新 Focus/Grab epoch
        epochResetThresholdSec (1,1) double {mustBePositive} = 0.05
    end

    properties (Hidden)
        % 图像 ring buffer
        buf int16 = int16([])   % H x W x N
        bufIdx (1,1) double = 0
        bufFilled (1,1) logical = false
        lastSaveT
        lastFrameSize

        % 与 buf 同索引的 ring buffers
        siTimeSecBuf double = []     % 1 x N，raw ScanImage timestamp（秒）
        monoTimeSecBuf double = []   % 1 x N，global monotonic（秒）
        frameIdxBuf double = []      % 1 x N，FrameIndex（每次 Focus/Grab 从 0 起）

        % FrameIndex（epoch 内）
        frameCounter (1,1) double = 0

        % 用于检测 Focus/Grab：上一帧 raw ScanImage timestamp
        lastRawSiT (1,1) double = NaN

        % 调试：只打印一次
        printedTimestampSource (1,1) logical = false
        printedGlobalTic (1,1) logical = false

        % 异步写盘状态（避免在 frameAcquired 回调里做重 I/O）
        saveFutures = []
        asyncSaveAvailable (1,1) logical = false
        warnedNoAsync (1,1) logical = false
    end

    methods
        function obj = RollingLastFramesSaver(varargin)
            % Optional name-value initialization so startup log reflects
            % caller config (instead of class defaults).
            if ~isempty(varargin)
                if mod(numel(varargin),2) ~= 0
                    error('RollingLastFramesSaver:InvalidArgs', ...
                        'Constructor expects name-value pairs.');
                end

                p = inputParser;
                p.FunctionName = 'RollingLastFramesSaver';
                addParameter(p, 'N', obj.N);
                addParameter(p, 'channelToUse', obj.channelToUse);
                addParameter(p, 'saveEverySec', obj.saveEverySec);
                addParameter(p, 'outDir', obj.outDir);
                addParameter(p, 'baseName', obj.baseName);
                addParameter(p, 'onlySaveWhenBufferFilled', obj.onlySaveWhenBufferFilled);
                addParameter(p, 'epochResetThresholdSec', obj.epochResetThresholdSec);
                parse(p, varargin{:});

                r = p.Results;
                obj.N = r.N;
                obj.channelToUse = r.channelToUse;
                obj.saveEverySec = r.saveEverySec;
                obj.outDir = char(r.outDir);
                obj.baseName = char(r.baseName);
                obj.onlySaveWhenBufferFilled = logical(r.onlySaveWhenBufferFilled);
                obj.epochResetThresholdSec = r.epochResetThresholdSec;
            end

            obj.hSI = evalin('base','hSI');
            if isempty(obj.hSI)
                error('未在 base workspace 找到 hSI。请确认 ScanImage 已启动。');
            end

            if ~exist(obj.outDir,'dir')
                mkdir(obj.outDir);
            end

            % 全局 monotonic tic（只初始化一次，永不按 Focus/Grab 重置）
            obj.ensureGlobalMonotonicTic();

            % 绑定 frameAcquired
            obj.listener_frameAcquired = addlistener( ...
                obj.hSI.hUserFunctions, ...
                'frameAcquired', ...
                @obj.onFrameAcquired);

            % 注册到 hSI（用于杀 ghost）
            try
                setappdata(obj.hSI, 'RollingLastFramesSaverHandle', obj);
            catch
            end

            obj.lastSaveT = tic;
            obj.asyncSaveAvailable = obj.detectAsyncSaveCapability();

            fprintf('RollingLastFramesSaver 已启动：CH%d，最近%d帧，每%.1f秒保存一次。\n', ...
                obj.channelToUse, obj.N, obj.saveEverySec);
            fprintf('输出目录：%s\n', obj.outDir);
            fprintf('TXT 列：[ScanImageTimeSec(raw), MonotonicSec(global), FrameIndex(reset per Focus/Grab)]\n');
            if obj.asyncSaveAvailable
                fprintf('Disk save mode: async (backgroundPool)\n');
            else
                fprintf(2, 'Disk save mode: sync fallback (no backgroundPool). Consider installing Parallel Computing Toolbox.\n');
            end
        end

        function delete(obj)
            try
                if ~isempty(obj.listener_frameAcquired) && isvalid(obj.listener_frameAcquired)
                    delete(obj.listener_frameAcquired);
                end
            catch
            end

            % 清理异步任务
            try
                if ~isempty(obj.saveFutures)
                    for i = 1:numel(obj.saveFutures)
                        try
                            f = obj.saveFutures(i);
                            if isprop(f, 'State')
                                st = char(f.State);
                                if ~(strcmpi(st,'finished') || strcmpi(st,'failed'))
                                    cancel(f);
                                end
                            end
                        catch
                        end
                    end
                end
            catch
            end

            try
                if ~isempty(obj.hSI) && isvalid(obj.hSI) && isappdata(obj.hSI,'RollingLastFramesSaverHandle')
                    h = getappdata(obj.hSI,'RollingLastFramesSaverHandle');
                    if isequal(h, obj)
                        rmappdata(obj.hSI,'RollingLastFramesSaverHandle');
                    end
                end
            catch
            end
        end

        function resetBuffer(obj)
            % 手动重置：清空 ring buffer + FrameIndex 从 0 开始
            obj.buf = int16([]);
            obj.bufIdx = 0;
            obj.bufFilled = false;
            obj.lastFrameSize = [];
            obj.lastSaveT = tic;

            obj.siTimeSecBuf = NaN(1,0);
            obj.monoTimeSecBuf = NaN(1,0);
            obj.frameIdxBuf = NaN(1,0);

            obj.frameCounter = 0;
            obj.lastRawSiT = NaN;
            obj.printedTimestampSource = false;

            fprintf('已重置缓冲区（FrameIndex 从 0 开始；Monotonic 仍为全局）。\n');
        end

        function printConfig(obj)
            fprintf('当前配置：CH%d，最近%d帧，每%.1f秒保存一次。\n', ...
                obj.channelToUse, obj.N, obj.saveEverySec);
            fprintf('输出目录：%s\n', obj.outDir);
            fprintf('文件前缀：%s\n', obj.baseName);
            fprintf('onlySaveWhenBufferFilled：%d\n', obj.onlySaveWhenBufferFilled);
            fprintf('epochResetThresholdSec：%.3f sec\n', obj.epochResetThresholdSec);
            fprintf('状态：bufFilled=%d, bufIdx=%d, frameCounter=%d, lastRawSiT=%g\n', ...
                obj.bufFilled, obj.bufIdx, obj.frameCounter, obj.lastRawSiT);

            if isempty(obj.buf)
                fprintf('图像缓冲区：未初始化（尚未收到第一帧）。\n');
            else
                fprintf('图像缓冲区：已初始化，尺寸=%dx%dx%d\n', ...
                    size(obj.buf,1), size(obj.buf,2), size(obj.buf,3));
            end
        end

        function onFrameAcquired(obj, ~, ~)
            % 1) 取最新 stripe
            stripe = [];
            try
                p = obj.hSI.hDisplay.stripeDataBufferPointer;
                stripe = obj.hSI.hDisplay.stripeDataBuffer{p};
            catch
                return
            end
            if isempty(stripe)
                return
            end

            % 2) 找到目标通道
            try
                channels = stripe.channelNumbers;
            catch
                return
            end
            f = find(channels == obj.channelToUse, 1, 'first');
            if isempty(f)
                return
            end

            % 3) 取 ROI1 图像
            frame = [];
            try
                frame = stripe.roiData{1}.imageData{f}{1};
            catch
                return
            end
            if isempty(frame)
                return
            end

            % 4) 转 int16
            if ~isa(frame,'int16')
                frame = int16(frame);
            end

            % 5) 初始化/重置缓冲区（尺寸变化或 N 变化）
            if isempty(obj.buf)
                obj.allocateBuffers(size(frame));
            else
                if ~isequal(size(frame), obj.lastFrameSize) || size(obj.buf,3) ~= obj.N
                    obj.allocateBuffers(size(frame));
                end
            end

            % 6) 读取 raw ScanImage timestamp（秒）
            rawSi = obj.getFrameTimestampSec(stripe);

            % 7) 检测 Focus/Grab：若发生回跳/归零，则清 buffer + FrameIndex=0
            if obj.shouldResetEpoch(rawSi)
                obj.resetForNewFocusGrab();
                % 注意：不重置 GLOBAL_MONO_TIC（保持全局对齐）
            end

            % 8) global monotonic
            tMono = obj.getMonotonicSecGlobal();

            % 9) 本帧 FrameIndex（epoch 内从 0 开始）
            frameIdxThis = obj.frameCounter;

            % 10) ring idx 前移并写入
            obj.bufIdx = obj.bufIdx + 1;
            if obj.bufIdx > obj.N
                obj.bufIdx = 1;
                obj.bufFilled = true;
            end

            obj.buf(:,:,obj.bufIdx) = frame;
            obj.siTimeSecBuf(1, obj.bufIdx)   = rawSi;
            obj.monoTimeSecBuf(1, obj.bufIdx) = tMono;
            obj.frameIdxBuf(1, obj.bufIdx)    = frameIdxThis;

            % 11) 更新检测状态（上一帧 raw timestamp）
            if ~isnan(rawSi)
                obj.lastRawSiT = rawSi;
            end

            % 12) FrameIndex 自增（下一帧）
            obj.frameCounter = obj.frameCounter + 1;

            % 调试：只打印一次
            if ~obj.printedTimestampSource
                fprintf('FrameTimestamp(raw) example (sec): %g\n', rawSi);
                fprintf('Monotonic(global) example (sec): %.6f\n', tMono);
                fprintf('FrameIndex example: %d\n', frameIdxThis);
                obj.printedTimestampSource = true;
            end

            % 13) 到时间就保存
            if toc(obj.lastSaveT) >= obj.saveEverySec

                if obj.onlySaveWhenBufferFilled && ~obj.bufFilled
                    obj.lastSaveT = tic;
                    return
                end

                obj.lastSaveT = tic;

                if obj.bufFilled
                    idxOrder = [obj.bufIdx+1:obj.N, 1:obj.bufIdx];
                else
                    idxOrder = 1:obj.bufIdx;
                end

                stack   = obj.buf(:,:,idxOrder);
                siSeq   = obj.siTimeSecBuf(1, idxOrder);
                monoSeq = obj.monoTimeSecBuf(1, idxOrder);
                frmSeq  = obj.frameIdxBuf(1, idxOrder);

                tsFile = datestr(now,'yyyymmdd_HHMMSS');
                outTif = fullfile(obj.outDir, sprintf('%s_%s.tif', obj.baseName, tsFile));
                outTxt = fullfile(obj.outDir, sprintf('%s_%s_time.txt', obj.baseName, tsFile));
                obj.dispatchSave(stack, siSeq, monoSeq, frmSeq, outTif, outTxt);
            end

            % 回调末尾轮询异步状态（轻量）
            obj.pollAsyncSaveState();
        end
    end

    methods (Hidden)
        function tf = detectAsyncSaveCapability(obj)
            tf = false;
            try
                if license('test','Distrib_Computing_Toolbox')
                    p = backgroundPool; %#ok<NASGU>
                    tf = true;
                end
            catch
                tf = false;
            end
        end

        function dispatchSave(obj, stack, siSeq, monoSeq, frmSeq, outTif, outTxt)
            if obj.asyncSaveAvailable
                try
                    f = parfeval( ...
                        backgroundPool, ...
                        @RollingLastFramesSaver.writeSnapshotToDisk, ...
                        0, ...
                        stack, siSeq, monoSeq, frmSeq, outTif, outTxt, obj.channelToUse);
                    if isempty(obj.saveFutures)
                        obj.saveFutures = f;
                    else
                        obj.saveFutures(end+1) = f; %#ok<AGROW>
                    end
                    return
                catch ME
                    if ~obj.warnedNoAsync
                        fprintf(2, '[RollingLastFramesSaver] Async save unavailable (%s). Fallback to sync save.\n', ME.message);
                        obj.warnedNoAsync = true;
                    end
                end
            end

            % fallback: 同步写盘（可能较慢）
            RollingLastFramesSaver.writeSnapshotToDisk(stack, siSeq, monoSeq, frmSeq, outTif, outTxt, obj.channelToUse);
        end

        function pollAsyncSaveState(obj)
            if isempty(obj.saveFutures)
                return
            end
            keep = [];
            for i = 1:numel(obj.saveFutures)
                f = obj.saveFutures(i);
                try
                    st = char(f.State);
                    if strcmpi(st,'finished')
                        % drop
                    elseif strcmpi(st,'failed')
                        msg = 'unknown async save error';
                        try
                            if ~isempty(f.Error)
                                msg = f.Error.message;
                            end
                        catch
                        end
                        fprintf(2, '[RollingLastFramesSaver] Async save failed: %s\n', msg);
                    else
                        if isempty(keep)
                            keep = f;
                        else
                            keep(end+1) = f; %#ok<AGROW>
                        end
                    end
                catch
                end
            end
            obj.saveFutures = keep;
            if ~isempty(obj.saveFutures) && numel(obj.saveFutures) > 4
                fprintf(2, '[RollingLastFramesSaver] Save queue backlog: %d pending tasks.\n', numel(obj.saveFutures));
            end
        end

        function allocateBuffers(obj, frameSize)
            obj.lastFrameSize = frameSize;
            obj.buf = zeros([frameSize, obj.N], 'int16');
            obj.bufIdx = 0;
            obj.bufFilled = false;

            obj.siTimeSecBuf   = NaN(1, obj.N);
            obj.monoTimeSecBuf = NaN(1, obj.N);
            obj.frameIdxBuf    = NaN(1, obj.N);
        end

        function rawSec = getFrameTimestampSec(obj, stripe)
            rawSec = NaN;
            try
                rawSec = stripe.frameTimestamp;
                return
            catch
            end
            try
                rawSec = obj.hSI.hScan2D.frameTimestamp;
            catch
                rawSec = NaN;
            end
        end

        function ensureGlobalMonotonicTic(obj)
            global GLOBAL_MONO_TIC
            if isempty(GLOBAL_MONO_TIC)
                GLOBAL_MONO_TIC = tic;
                if ~obj.printedGlobalTic
                    fprintf('Global monotonic tic initialized.\n');
                    obj.printedGlobalTic = true;
                end
            end
        end

        function tMonoSec = getMonotonicSecGlobal(obj)
            global GLOBAL_MONO_TIC
            if isempty(GLOBAL_MONO_TIC)
                GLOBAL_MONO_TIC = tic; % 兜底
            end
            tMonoSec = toc(GLOBAL_MONO_TIC);
        end

        function tf = shouldResetEpoch(obj, rawSi)
            % 判定是否进入新 Focus/Grab epoch（通过 raw frameTimestamp 回跳/归零）
            tf = false;

            if isnan(rawSi) || isnan(obj.lastRawSiT)
                return
            end

            % 1) 明显回跳
            if rawSi < (obj.lastRawSiT - obj.epochResetThresholdSec)
                tf = true;
                return
            end

            % 2) 归零特征：上一帧已跑一段，但当前帧接近 0
            if (obj.lastRawSiT > 0.5) && (rawSi <= obj.epochResetThresholdSec)
                tf = true;
                return
            end
        end

        function resetForNewFocusGrab(obj)
            % 新 Focus/Grab：清 ring buffer 状态 + FrameIndex=0
            obj.frameCounter = 0;
            obj.lastRawSiT = NaN;

            obj.bufIdx = 0;
            obj.bufFilled = false;

            if ~isempty(obj.siTimeSecBuf),   obj.siTimeSecBuf(:) = NaN; end
            if ~isempty(obj.monoTimeSecBuf), obj.monoTimeSecBuf(:) = NaN; end
            if ~isempty(obj.frameIdxBuf),    obj.frameIdxBuf(:) = NaN; end

            fprintf('=== Detected new Focus/Grab: cleared buffer + FrameIndex reset to 0 (Monotonic remains global) ===\n');
        end
    end

    methods (Static, Hidden)
        function writeSnapshotToDisk(stack, siSeq, monoSeq, frmSeq, outTif, outTxt, channelToUse)
            RollingLastFramesSaver.writeTiffStackInt16(stack, outTif);

            fid = -1;
            try
                fid = fopen(outTxt, 'w');
                if fid == -1
                    error('Cannot open txt file for writing.');
                end

                fprintf(fid, 'ScanImageTimeSec\tMonotonicSec\tFrameIndex\n');
                for k = 1:numel(frmSeq)
                    fprintf(fid, '%.6f\t%.6f\t%d\n', siSeq(k), monoSeq(k), frmSeq(k));
                end
            catch
            end
            if fid ~= -1
                fclose(fid);
            end

            fprintf('已保存 CH%d 最近 %d 帧：%s\n', channelToUse, size(stack,3), outTif);
            fprintf('已保存 Time TXT（%d 行）：%s\n', numel(frmSeq), outTxt);
        end

        function writeTiffStackInt16(stack, filename)
            [H, W, N] = size(stack);

            t = Tiff(filename, 'w');

            tagstruct.ImageLength = H;
            tagstruct.ImageWidth  = W;
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.BitsPerSample = 16;
            tagstruct.SamplesPerPixel = 1;
            tagstruct.SampleFormat = Tiff.SampleFormat.Int;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.RowsPerStrip = H;
            tagstruct.Compression = Tiff.Compression.None;

            for k = 1:N
                t.setTag(tagstruct);
                t.write(stack(:,:,k));

                if k < N
                    t.writeDirectory();
                end
            end

            t.close();
        end
    end
end
