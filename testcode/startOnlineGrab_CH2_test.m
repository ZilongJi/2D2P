function S = startOnlineGrab_CH2(varargin)
% startOnlineGrab_CH2
% 每次调用都强制重启 Online Grab：
%  - 从 hSI 的 appdata 里杀掉 ghost RollingLastFramesSaver
%  - 不再全局清理 instrument/serial 对象，避免误关正在运行的 rotary logger
%  - 删除 base workspace 里的旧 S
%  - 创建新 RollingLastFramesSaver，应用配置，resetBuffer
%
% REQUIREMENT:
%   You MUST pass outDir explicitly:
%       S = startOnlineGrab_CH2('outDir', 'D:\path\to\session_folder');
%
% 用法：
%   S = startOnlineGrab_CH2('outDir', outDir);

    % ===== parse required outDir (MUST be provided) =====
    if nargin < 2
        error('startOnlineGrab_CH2:MissingOutDir', ...
              'You must call startOnlineGrab_CH2(''outDir'', <folder>).');
    end

    if ~(ischar(varargin{1}) || isstring(varargin{1})) || ...
       ~strcmpi(char(varargin{1}), 'outDir')
        error('startOnlineGrab_CH2:InvalidSyntax', ...
              'Expected first argument ''outDir''. Example: startOnlineGrab_CH2(''outDir'',''D:\\data\\session1'')');
    end

    outDir = varargin{2};

    if ~(ischar(outDir) || isstring(outDir))
        error('startOnlineGrab_CH2:InvalidOutDir', ...
              'outDir must be a char or string.');
    end

    outDir = char(outDir);

    if ~exist(outDir,'dir')
        mkdir(outDir);
    end
    % ===================================================

    % ====== 用户配置区（只需改这里） ======
    cfg.channelToUse = 2;       % CH2
    cfg.N = 100;                % 最近 N 帧
    cfg.saveEverySec = 30;      % 每隔多少秒保存一次
    cfg.outDir = outDir;        % <-- REQUIRED from caller
    cfg.baseName = 'online_grab_CH2';
    % 必须按时间窗口稳定产出文件给下游分析，因此不要求先填满 N 帧
    cfg.onlySaveWhenBufferFilled = false;
    % =======================================

    % 0) 杀掉 hSI appdata 中记录的 ghost（最可靠）
    try
        hSI = evalin('base','hSI');
        if ~isempty(hSI) && isvalid(hSI) && isappdata(hSI,'RollingLastFramesSaverHandle')
            old = getappdata(hSI,'RollingLastFramesSaverHandle');
            try, delete(old); catch, end
            try, rmappdata(hSI,'RollingLastFramesSaverHandle'); catch, end
            fprintf('Deleted ghost RollingLastFramesSaverHandle from hSI.\n');
        end
    catch
    end

    % 1) IMPORTANT:
    %    这里不要全局清理 instrfind 对象。
    %    因为 startRotaryLogger 会保持一个打开的 serial（例如 COM7），
    %    如果这里 fclose/delete 全部 instrument，会把 rotary 的串口也关掉，
    %    然后 rotary 定时器读角度就会变成 NaN。
    %
    %    如果确实需要清理串口，请在 rotary logger 未运行时手动执行。

    % 2) 强制清理 base workspace 里的旧对象 S
    try
        if evalin('base','exist(''S'',''var'')')
            Sold = evalin('base','S');
            try, delete(Sold); catch, end
            evalin('base','clear S');
        end
    catch
    end

    % 3) 创建新对象（在构造时就传入配置，确保启动日志即为目标参数）
    S = RollingLastFramesSaver( ...
        'channelToUse', cfg.channelToUse, ...
        'N', cfg.N, ...
        'saveEverySec', cfg.saveEverySec, ...
        'outDir', cfg.outDir, ...
        'baseName', cfg.baseName, ...
        'onlySaveWhenBufferFilled', cfg.onlySaveWhenBufferFilled);

    % 4) 重置 buffer + 打印确认
    S.resetBuffer();
    S.printConfig();

    % 5) 放回 base workspace（便于你随时手动 delete(S) 或查看状态）
    assignin('base','S',S);

    fprintf('=== Online Grab 已重启：现在可以 Focus / Grab ===\n');
end
