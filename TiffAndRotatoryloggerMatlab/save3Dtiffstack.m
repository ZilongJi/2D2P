function save3Dtiffstack(stack, filename)
%save3Dtiffstack Write a 3-D int16 array as a multipage TIFF
%
%   save3Dtiffstack(stack, filename)
%
%   stack    : H × W × N int16 array
%   filename : output TIFF file

    arguments
        stack (:,:,:) int16
        filename (1,:) char
    end

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
