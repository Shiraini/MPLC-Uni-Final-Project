import numpy as np
import copy
import matplotlib.pyplot as plt

class MPLCSystem:
    def __init__(self, planes, d, lr):
        self.planes = planes
        self.n_planes = len(planes)
        self.lr = lr
        self.T = None
        self._set_d_list(d)   # supports scalar or list of length n_planes+1

    # --- distances handling ---
    def _set_d_list(self, d):
        if np.isscalar(d):
            self.d_list = [float(d)] * (self.n_planes + 1)
        else:
            d = list(d)
            if len(d) != self.n_planes + 1:
                raise ValueError(f"d must be scalar or length {self.n_planes+1}")
            self.d_list = [float(x) for x in d]
        # keep legacy scalar for any external reads; matches your previous meaning
        self.d = self.d_list[0]

    def set_distances(self, d):
        """Set scalar d or list/array of length n_planes+1 (pre, between..., post)."""
        self._set_d_list(d)

    # --- SNAPSHOTS EXACTLY AS BEFORE ---
    def forward_propagate(self, mode):
        """
        Original semantics:
        propagate(d) -> apply -> SNAPSHOT -> propagate(d) ... (repeat)
        Now with per-segment distances: d_list[0], d_list[1], ..., d_list[-1]
        """
        field_after = []
        # pre-distance before plane 0
        mode.propagate(self.d_list[0])
        for i, plane in enumerate(self.planes):
            field_after.append(mode.field.copy())
            plane.apply(mode)
            # field_after.append(mode.field.copy())
            mode.propagate(self.d_list[i+1])
        return field_after

    def backward_propagate(self, mode):
        """
        Original semantics:
        propagate(-d) -> SNAPSHOT -> apply(back=True) -> propagate(-d) ... (reverse loop)
        Return reversed list to align indices with forward_propagate.
        """
        field_before = []
        # start from output: go back by final segment
        mode.propagate(-self.d_list[-1])
        for i, plane in enumerate(reversed(self.planes)):
            # field_before.append(mode.field.copy())
            plane.apply(mode, back=True)
            field_before.append(mode.field.copy())

            seg = self.d_list[-(i+2)]
            mode.propagate(-seg)
        return field_before[::-1]

    def sort(self, mode, record=False, steps_per_propagation=12):
        """
        Original snapshot timing with segmented propagation:
        [init] -> (prop d, snapshot segments) -> apply, snapshot -> ... -> (final prop d, snapshot segments)
        """
        snaps = []
        if record:
            snaps.append(mode.field.copy())                     # initial input

        # pre-segment before plane 0
        for seg in mode.propagate_segmented(self.d_list[0], steps=steps_per_propagation):
            if record:
                snaps.append(seg)

        for i, plane in enumerate(self.planes):
            plane.apply(mode)
            if record:
                snaps.append(mode.field.copy())                 # AFTER apply (per plane)
            for seg in mode.propagate_segmented(self.d_list[i+1], steps=steps_per_propagation):
                if record:
                    snaps.append(seg)

        return snaps if record else None

    # --- training / metrics (unchanged) ---
    def fit_fontaine(self, inputs, targets, iterations=10):
        for _ in range(iterations):
            masks = []
            inputs = inputs.copy()
            targets = targets.copy()

            for input_mode_orig, target_mode_orig in zip(inputs, targets):
                input_mode = copy.deepcopy(input_mode_orig)
                target_mode = copy.deepcopy(target_mode_orig)

                fwd_fields = self.forward_propagate(input_mode)     # AFTER-apply snapshots
                bwd_fields = self.backward_propagate(target_mode)   # BEFORE-back-apply (aligned)

                per_mode = [b * np.conj(f) for b, f in zip(bwd_fields, fwd_fields)]
                masks.append(np.stack(per_mode))  # (num_planes, Ny, Nx)

            total_inner = np.sum(masks, axis=0)
            for i, plane in enumerate(self.planes):
                plane.phase = np.mod(plane.phase + self.lr * np.angle(total_inner[i]), 2*np.pi)

    def compute_transfer_matrix(self, inputs, targets):
        N_in = len(inputs); N_out = len(targets)
        T = np.zeros((N_out, N_in), dtype=complex)
        for i, input_mode in enumerate(inputs):
            mode_copy = copy.deepcopy(input_mode)
            self.sort(mode_copy)  # propagate through trained MPLC
            output_field = mode_copy.field
            for j, target_mode in enumerate(targets):
                tgt_field = target_mode.field
                norm_tgt = tgt_field / np.linalg.norm(tgt_field)
                norm_out = output_field / np.linalg.norm(output_field)
                T[j, i] = np.vdot(norm_tgt, norm_out)
        self.T = T

    def compute_IL_MDL_from_T(self):
        if self.T is None:
            print('Error! please compute T')
            return None, None
        power_per_input = np.sum(np.abs(self.T)**2, axis=0)
        IL  = -10*np.log10(np.mean(power_per_input))
        MDL =  10*np.log10(np.max(power_per_input)/np.min(power_per_input))
        return IL, MDL

    def visualize_crosstalk_matrix(self, input_labels=None, target_labels=None, title="Crosstalk Matrix"):
        T = self.T
        power_matrix = np.abs(T)**2
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(power_matrix, cmap='viridis', interpolation='nearest')
        ax.set_title(title); ax.set_xlabel("Input Mode Index"); ax.set_ylabel("Target Mode Index")
        if input_labels:
            ax.set_xticks(range(len(input_labels))); ax.set_xticklabels(input_labels, rotation=45)
        if target_labels:
            ax.set_yticks(range(len(target_labels))); ax.set_yticklabels(target_labels)
        plt.colorbar(im, ax=ax, label='Power Coupling |T|²')
        plt.tight_layout(); plt.show()
