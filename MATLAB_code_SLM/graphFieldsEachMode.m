figure(2);
for planeIdx=1:planeCount
    TOTAL = single(0);
    for modeIdx=1:modeCount
        TOTAL = TOTAL+squeeze(abs(FIELDS(1,planeIdx,modeIdx,:,:)).^2);
    end
    
    TOTAL2 = single(0);
    %for modeIdx=1:modeCount
    for modeIdx=2:2
        TOTAL2 = TOTAL2+squeeze(abs(FIELDS(2,planeIdx,modeIdx,:,:)).^2);
    end
    
    %Sum of all mode intensities in the forward direction
    subplot(3,planeCount,planeIdx);
    imagesc(TOTAL);
    axis equal;
    axis off;
    
    %Sum of all mode intensities in the backward direction
    subplot(3,planeCount,planeIdx+planeCount);
    imagesc(TOTAL2);
    axis equal;
    axis off;
    
    %Angle of the masks
    subplot(3,planeCount,planeIdx+2.*planeCount);
    imagesc(squeeze(angle(MASKS(planeIdx,:,:))));
    axis equal;
    axis off;
    set(gca,'CLim',[-pi pi]);
end
