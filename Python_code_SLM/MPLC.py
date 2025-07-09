import numpy as np
import copy
import matplotlib.pyplot as plt

class MPLCSystem:


    def __init__(self, planes, d, lr):
        self.planes = planes
        self.n_planes = len(planes)
        self.d = d
        self.lr = lr
        self.T = None
        # Distance between planes

    def forward_propagate(self, mode):
        """Propagate a mode forward through all planes, storing fields before each mask."""
        field_before = []
        mode.propagate(self.d)
        for plane in self.planes:
            plane.apply(mode)
            field_before.append(mode.field.copy())
            mode.propagate(self.d)
        return field_before


    def backward_propagate(self, mode):
        """Propagate a mode backward through all planes, storing fields after each inverse mask."""
        field_after = []
        mode.propagate(-self.d)
        for plane in reversed(self.planes):
            #switch back to after applying
            field_after.append(mode.field.copy())
            plane.apply(mode, back=True)
            mode.propagate(-self.d)
        return field_after[::-1]  # Reverse to match forward order


    def fit_fontaine(self, inputs, targets, iterations=10):
        """Run iterative optimization to find phase masks using the Fontaine algorithm."""
        for _ in range(iterations):
            masks = []
            inputs = inputs.copy()
            targets = targets.copy()

            for input_mode_orig, target_mode_orig in zip(inputs, targets):
                input_mode = copy.deepcopy(input_mode_orig)
                target_mode = copy.deepcopy(target_mode_orig)
                # Forward propagation
                fwd_fields = self.forward_propagate(input_mode)
                # Backward propagation
                bwd_fields = self.backward_propagate(target_mode)

                # Compute per-mode inner product per plane
                per_mode = [b * np.conj(f) for b, f in zip(bwd_fields, fwd_fields)]
                masks.append(np.stack(per_mode))  # Shape: (num_planes, Ny, Nx)

            # Sum contributions from all modes
            total_inner = np.sum(masks, axis=0)  # Shape: (num_planes, Ny, Nx)

            # Update phase masks
            for i, plane in enumerate(self.planes):
                plane.phase = np.mod(plane.phase +
                    self.lr * np.angle(total_inner[i]),
                    2 * np.pi
                )


    def sort(self, mode, record=False, steps_per_propagation=12):
        d = self.d
        snapshots = []

        if record:
            snapshots.append(mode.field.copy())  # Initial input

        for plane in self.planes:
            # Propagate in mini-steps
            for segment in mode.propagate_segmented(d, steps=steps_per_propagation):
                if record:
                    snapshots.append(segment)

            # Apply mask
            plane.apply(mode)
            if record:
                snapshots.append(mode.field.copy())

        # Final segment after last mask
        for segment in mode.propagate_segmented(d, steps=steps_per_propagation):
            if record:
                snapshots.append(segment)

        return snapshots if record else None

    def compute_transfer_matrix(self, inputs, targets):
        N_in = len(inputs)
        N_out = len(targets)
        T = np.zeros((N_out, N_in), dtype=complex)

        for i, input_mode in enumerate(inputs):
            mode_copy = copy.deepcopy(input_mode)
            self.sort(mode_copy)  # propagate through trained MPLC

            output_field = mode_copy.field

            for j, target_mode in enumerate(targets):
                # Normalize both fields to unit power before inner product
                tgt_field = target_mode.field
                norm_tgt = tgt_field / np.linalg.norm(tgt_field)
                norm_out = output_field / np.linalg.norm(output_field)

                T[j, i] = np.vdot(norm_tgt, norm_out)  # inner product ⟨target|output⟩

        self.T = T

    def compute_IL_MDL_from_T(self):
        if self.T is None:
            print('Error! please compute T')
        else:
            power_per_input = np.sum(np.abs(self.T) ** 2, axis=0)  # shape: (N_inputs,)

            IL = -10 * np.log10(np.mean(power_per_input))
            MDL = 10 * np.log10(np.max(power_per_input) / np.min(power_per_input))

        return IL, MDL

    import matplotlib.pyplot as plt

    def visualize_crosstalk_matrix(self, input_labels=None, target_labels=None, title="Crosstalk Matrix"):
        T = self.T
        power_matrix = np.abs(T) ** 2

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(power_matrix, cmap='viridis', interpolation='nearest')

        ax.set_title(title)
        ax.set_xlabel("Input Mode Index")
        ax.set_ylabel("Target Mode Index")

        if input_labels:
            ax.set_xticks(range(len(input_labels)))
            ax.set_xticklabels(input_labels, rotation=45)

        if target_labels:
            ax.set_yticks(range(len(target_labels)))
            ax.set_yticklabels(target_labels)

        plt.colorbar(im, ax=ax, label='Power Coupling |T|²')
        plt.tight_layout()
        plt.show()


