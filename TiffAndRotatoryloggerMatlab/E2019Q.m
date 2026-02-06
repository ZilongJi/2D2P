classdef E2019Q
    methods(Static)
        %%%%%%%%%%%%%%%% OPEN/CLOSE COM port functions %%%%%%%%%%%%%%%%%%%%
        % Open COM  Port for E201-9Q
        function FID = Open_COM_Port(ComString)
            FID = serial(ComString);
            FID.Terminator = '';
            % 建议：适当增大输入缓冲，减少溢出风险
            try
                FID.InputBufferSize = 65536;
            catch
            end
            fopen(FID);
        end
        
        % Close COM Port for E201-9Q
        function Close_COM_Port(FID)
            fclose(FID);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%% Status functions %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Read software version of E201-9Q
        function data = GetSoftwareVersion(FID)
            fprintf(FID,'v');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Read serial number of E201-9Q
        function data = GetSerialNumber(FID)
            fprintf(FID,'s');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Read encoder supply status, voltage and current consumption
        function data = GetEncSupply(FID)
            fprintf(FID,'e');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Read status of hardware input pins on interface
        function data = GetInputPinStatus(FID)
            fprintf(FID,'p');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%% Power management functions %%%%%%%%%%%%%%%%%%%%%
        % Turn ON power supply to encoder
        function data = EncSupply_ON(FID)
            fprintf(FID,'n');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Turn OFF power supply to encoder
        function data = EncSupply_OFF(FID)
            fprintf(FID,'f');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%% Functions related to the encoder position %%%%%%%%%%%%
        % Read encoder position (string, decimal)
        function data = GetEncPosition(FID)
            fprintf(FID,'?');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Read encoder position (string, decimal) with timestamp
        function data = GetEncPosition_Timestamp(FID)
            fprintf(FID,'!');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Read encoder position (string, HEX)
        function data = GetEncPositionHEX(FID)
            fprintf(FID,'>');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Read encoder position (string, HEX) with timestamp
        function data = GetEncPositionHEX_Timestamp(FID)
            fprintf(FID,'<');
            data = [];
            start = clock;
            while(isempty(strfind(data, 13)))
                if FID.BytesAvailable > 0
                    data = [data fscanf(FID,'%c',FID.BytesAvailable)]; %#ok<AGROW>
                end
                if etime(clock,start) > 3
                    disp('Timeout occurs while reading COM port');
                    break
                end
            end
        end
        
        % Clear reference status flag
        function ClearReferenceFlag(FID)
            fprintf(FID,'c');
        end
        
        % Set current count value to zero (also affects reference mark)
        function ResetCurrentCount(FID)
            fprintf(FID,'z');
        end
        
        % Clear zero offset value stored by "ResetCurrentCount" function
        function ClearZeroOffset(FID)
            fprintf(FID,'a');
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%% String to Double converting functions %%%%%%%%%%%%%
        % Get encoder count in double precision format (LEGACY)
        function data = GetEncCountDOUBLE(FID)
            temp = E2019Q.GetEncPosition(FID);
            data = str2double(temp(2:min(strfind(temp,':')-1)));
        end
        
        % Get encoder reference mark in double precision format (LEGACY)
        function data = GetEncReferenceDOUBLE(FID)
            temp = E2019Q.GetEncPosition(FID);
            data = str2double(temp(min(strfind(temp,':'))+2:max(strfind(temp,':'))-1));
        end
        
        % Get timestamp of position in double precision format (LEGACY)
        function data = GetTimestampDOUBLE(FID)
            temp = E2019Q.GetEncPosition_Timestamp(FID);
            data = str2double(temp(max(strfind(temp,':'))+2:end-1));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%% FAST FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get encoder count via FAST path (uint8 fread + CR scan + manual parse)
        %
        % 使用方法：
        %   c = E2019Q.GetEncCountFAST(FID);   % 返回 count（double）
        %
        function data = GetEncCountFAST(FID)
            % 说明：为保证速度，这里使用 persistent 缓冲。
            % 如果你担心多个串口实例并存，可改为 containers.Map 以 FID.Port 区分。
            persistent buf
            if isempty(buf)
                buf = uint8([]);
            end

            % 可选：尽量把 Timeout 调小（不改也行）
            % FID.Timeout = 0.02;  % 你也可以在外部脚本设置

            % 1) 发送 '?'（单字节，比 fprintf 更轻）
            fwrite(FID, uint8('?'), 'uint8');

            % 2) 读一行（以 CR=13 结束，不含 CR）
            [lineBytes, buf, ok] = E2019Q.readLineCR_uint8(FID, buf);
            if ~ok
                data = NaN;
                return;
            end

            % 3) 解析 count（按原逻辑：temp(2 : firstColon-1)）
            c = E2019Q.parseCountFast(lineBytes);
            data = double(c);
        end
        
    end

    methods(Static, Access=private)
        function [lineBytes, buf, ok] = readLineCR_uint8(FID, buf)
            % 从 FID 读到 CR(13) 为止。
            % buf: 持久缓冲，存储未消费字节（比如上次多读的部分）。
            CR = uint8(13);
            ok = true;
            lineBytes = uint8([]);

            % 先在 buf 里找 CR
            k = find(buf == CR, 1, 'first');
            if ~isempty(k)
                lineBytes = buf(1:k-1);
                buf = buf(k+1:end);
                return;
            end

            % buf 没 CR：继续读直到 Timeout
            t0 = tic;
            timeout = FID.Timeout;

            while true
                nb = FID.BytesAvailable;
                if nb > 0
                    newBytes = fread(FID, nb, 'uint8')';
                    if ~isempty(newBytes)
                        buf = [buf, newBytes]; %#ok<AGROW>
                        k = find(buf == CR, 1, 'first');
                        if ~isempty(k)
                            lineBytes = buf(1:k-1);
                            buf = buf(k+1:end);
                            return;
                        end
                    end
                else
                    if toc(t0) > timeout
                        ok = false;
                        return;
                    end
                    pause(0); % 让步，避免忙等占满CPU
                end
            end
        end

        function c = parseCountFast(lineBytes)
            % 解析 lineBytes(2 : before first ':') 的十进制整数
            COLON = uint8(':');
            n = numel(lineBytes);
            if n < 3
                c = NaN; 
                return;
            end

            k = find(lineBytes == COLON, 1, 'first');
            if isempty(k) || k <= 2
                c = NaN; 
                return;
            end

            s = lineBytes(2:k-1);

            % 符号
            idx = 1;
            sign = 1;
            if ~isempty(s) && s(1) == uint8('-')
                sign = -1; idx = 2;
            elseif ~isempty(s) && s(1) == uint8('+')
                idx = 2;
            end

            val = 0;
            for i = idx:numel(s)
                d = s(i);
                if d < uint8('0') || d > uint8('9')
                    break;
                end
                val = val * 10 + double(d - uint8('0'));
            end
            c = sign * val;
        end
    end
end
