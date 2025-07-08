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
            field_before.append(mode.field.copy())
            plane.apply(mode)
            mode.propagate(self.d)
        return field_before


    def backward_propagate(self, mode):
        """Propagate a mode backward through all planes, storing fields after each inverse mask."""
        field_after = []
        mode.propagate(-self.d)
        for plane in reversed(self.planes):
            #switch back to after applying
            plane.apply(mode, back=True)
            field_after.append(mode.field.copy())
            mode.propagate(-self.d)
        return field_after[::-1]  # Reverse to match forward order

    def replace_phase(self, patient, donor):
        """Replace the phase of 'patient' with the phase of 'donor'."""
        amp = np.abs(patient.field)
        phase = np.angle(donor.field)
        patient.field = amp * np.exp(1j * phase)
        return patient

    def TG(self, inputs, targets, iterations=10):
        for i in range(iterations):
            self.train_plane(inputs, targets, i % len(self.planes))

    def fwd_field_at_plane(self, input, plane):
        d = self.d
        input.propagate(d)  # first free space
        for i in range(plane):
            self.planes[i].apply(input)
            input.propagate(d)
        return input.field

    def bwd_field_at_plane(self, target, plane):
        d = self.d
        target.propagate(-d)  # first free space
        for i in reversed(range(plane + 1, len(self.planes))):
            self.planes[i].apply(target, back=True)
            target.propagate(-d)
        self.planes[plane].apply(target, back=True)
        return target.field

    def train_plane(self, inputs, targets, plane_numer):
        masks = []
        # propagate fwd to plane
        for input, target in zip(inputs, targets):
            input_copy = copy.deepcopy(input)
            target_copy = copy.deepcopy(target)
            field_before = self.fwd_field_at_plane(input_copy, plane_numer)
            field_after = self.bwd_field_at_plane(target_copy, plane_numer)
            newphase = np.angle(field_before*np.conj(field_after)) ##maybe minus
            masks.append(newphase)
        phase = np.angle(np.sum(masks, axis=0))
        self.planes[plane_numer].phase = np.mod(
            self.planes[plane_numer].phase + self.lr * phase, 2 * np.pi
        )

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

                # Phase transplant at output
                #self.replace_phase(target_mode, input_mode)

                # Backward propagation
                bwd_fields = self.backward_propagate(target_mode)

                # Compute per-mode inner product per plane
                per_mode = [b * np.conj(f) for b, f in zip(bwd_fields, fwd_fields)]
                masks.append(np.stack(per_mode))  # Shape: (num_planes, Ny, Nx)

            # Sum contributions from all modes
            total_inner = np.sum(masks, axis=0)  # Shape: (num_planes, Ny, Nx)

            # Update phase masks
            for i, plane in enumerate(self.planes):
                plane.phase = np.mod(
                    plane.phase + self.lr * np.angle(total_inner[i]),
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

    def pre_train(self, inputs, targets, plane_numer):
        masks = []
        # propagate fwd to plane
        for input, target in zip(inputs, targets):
            input_copy = copy.deepcopy(input)
            target_copy = copy.deepcopy(target)
            field_before = self.fwd_field_at_plane(input_copy, plane_numer)
            field_after = self.bwd_field_at_plane(target_copy, plane_numer)
            newphase = np.angle(field_before * np.conj(field_after))  ##maybe minus
            masks.append(newphase)
        phase = np.angle(np.sum(masks, axis=0))
        self.planes[plane_numer].phase = phase

    def pre_TG(self, inputs, targets):
        for i in range(self.n_planes):
            self.pre_train(inputs, targets, i)

    def measure_losses(self, inputs, targets):
        eta_list = []

        for input_mode, target_mode in zip(inputs, targets):
            input_copy = copy.deepcopy(input_mode)
            target_copy = copy.deepcopy(target_mode)

            self.sort(input_copy)  # applies propagation + masks
            overlap = np.vdot(target_copy.field, input_copy.field)
            eta = np.abs(overlap) ** 2
            eta_list.append(eta)

        eta_array = np.array(eta_list)

        IL = -10 * np.log10(np.mean(eta_array))
        MDL = 10 * np.log10(np.max(eta_array) / np.min(eta_array))

        return IL, MDL, eta_array  # Return all for plotting etc.

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


