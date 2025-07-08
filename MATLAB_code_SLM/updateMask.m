%The phase of the current mask in this plane
MASK = exp(1i.*angle(squeeze(MASKS(planeIdx,:,:))));
%What will become the new mask
MSK = 0;
%For every mode..
for modeIdx=1:modeCount
    %Field in the forward direction
    fieldForward  = (squeeze(FIELDS(1,planeIdx,modeIdx,:,:)));
    %Conjugate of the field in the backward direction
    fieldBackward = conj(squeeze(FIELDS(2,planeIdx,modeIdx,:,:)));
    %Calculate the total power of the field in forward and backward
    %directions. Used for normalizing, particularly if the k-space filter
    %is discarding a lot of power.
    pwrForward = sum(sum(abs(fieldForward).^2));
    pwrBackward = sum(sum(abs(fieldBackward).^2));
    
    %The spatial overlap of the forward and backward fields normalized to
    %their total power
    dMASK = (fieldForward.*fieldBackward)./sqrt(pwrForward.*pwrBackward);
    %Remember the coupling between forward and backward (total power of
    %overlapped fields, dMask)
    coupling(i,modeIdx) = abs(sum(sum(dMASK))).^2;
    %Take the overlap of our new mask component dMASK with the old MASK.
    dPhi = sum(sum(dMASK.*conj(MASK)));
    %boost can be changed to adjust the weight of each modal component to
    %the superposition used to make the mask. Can be used to minimise
    %differences in coupling between modes. Particularly relevant the more
    %modes there are, as each individual mode contributes only a small
    %amount to the overall insertion loss of the device, hence it becomes
    %easy for 1 mode to be neglected for relatively minor improvements to
    %all the other modes.
    boost = 1;
    %Try to add this dMASK component so that it constructively interfers
    %with whatever the existing MASK is.
    MSK = MSK+dMASK.*exp(-1i.*angle(dPhi)).*boost;
end

%If the symmetry constraint is in place, force the mask to be symmetric
if (symmetricMasks)
    MSK = (MSK+MSK(end:-1:1,:))./2.0;
end

%Store this mask, plus the maskOffset which is used to discourage
%phase-matching of low-intensity (scattered) light, and promote smooth
%solutions.
MASKS(planeIdx,:,:) = (MSK)+maskOffset;


