import re
from operator import add


class DataParser:

    """Class for reading relevant data from Gaussian output files into python dictionaries."""

    def __init__(self, file_path):

        """Constructor

        Args:
            file_path (string): Path to the output file.
        """

        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.lines = f.read().split('\n')
        self.n_atoms = self._get_number_of_atoms()

    def _get_number_of_atoms(self):

        for i in range(len(self.lines)):
            # find line that contain atom number
            if 'NAtoms=' in self.lines[i]:

                # get position in line
                line_split = self.lines[i].split()
                n_atomsIndex = line_split.index('NAtoms=') + 1

                # return
                return int(line_split[n_atomsIndex])

        raise Exception('Could not find number of atoms in file.')

    def parse(self):

        """Iterates through a given Gaussian output file and returns a dict of extracted data.

        Returns:
            dict[]: A dict containing different information extracted from the Gaussian output file.
        """

        # output variable
        qm_data = {}
        qm_data['n_atoms'] = self.n_atoms

        # get id token from file name
        qm_data['id'] = ''.join(self.file_path.split('/')[-1].split('.')[0:-1])

        # variable that shows if scan is currently in SVP or TZVP region of output file
        region_state = ''
        for i in range(len(self.lines)):

            if 'def2SVP' in self.lines[i]:
                region_state = 'svp'
            elif 'def2TZVP' in self.lines[i]:
                region_state = 'tzvp'

            # search for keywords and if found call appropriate functions with start index
            # the start index addition offset is based on the Gaussian output format
            if 'Standard orientation' in self.lines[i]:
                qm_data['atomic_numbers'] = self._extract_atomic_numbers(i + 5)
                qm_data['geometric_data'], i = self._extract_geometric_data(i + 5)

            if 'Summary of Natural Population Analysis' in self.lines[i]:
                qm_data['natural_atomic_charges'], qm_data['natural_electron_population'], i = self._extract_natural_atomic_charges(i + 6)

            if 'Natural Electron Configuration' in self.lines[i]:
                qm_data['natural_electron_configuration'], i = self._extract_natural_electron_configuration(i + 2)

            if 'Wiberg bond index matrix' in self.lines[i]:
                qm_data['wiberg_bond_order_matrix'], i = self._extract_index_matrix(i + 4)

            if 'Atom-Atom Net Linear NLMO/NPA' in self.lines[i]:
                qm_data['nlmo_bond_order_matrix'], i = self._extract_index_matrix(i + 4)

            if 'Bond orbital / Coefficients / Hybrids' in self.lines[i]:
                nbo_entries, i = self._extract_nbo_data(i + 2)

            if 'NATURAL BOND ORBITALS' in self.lines[i]:
                nbo_energies, i = self._extract_nbo_energies(i + 7)

            if '3-Center, 4-Electron A:-B-:C Hyperbonds (A-B :C <=> A: B-C)' in self.lines[i]:
                qm_data['hyper_node_data'] = self._extract_hyper_node_data(i + 7)

            if 'SECOND ORDER PERTURBATION THEORY ANALYSIS' in self.lines[i]:
                qm_data['sopa_data'], i = self._extract_sopa_data(i + 8)

            if 'Atom I' in self.lines[i]:
                qm_data['lmo_bond_order_matrix'], i = self._extract_lmo_bond_data(i + 1)

            if 'Charge = ' in self.lines[i]:
                qm_data['charge'] = self._extract_charge(i)

            if 'Stoichiometry' in self.lines[i]:
                qm_data['stoichiometry'] = self._extract_stoichiometry(i)

            if 'Molecular mass' in self.lines[i]:
                qm_data['molecular_mass'] = self._extract_molecular_mass(i)

            if 'Grimme-D3(BJ) Dispersion energy=' in self.lines[i]:
                if region_state == 'svp':
                    qm_data['svp_dispersion_energy'] = self._extract_dispersion_energy(i)
                elif region_state == 'tzvp':
                    qm_data['tzvp_dispersion_energy'] = self._extract_dispersion_energy(i)

            if 'SCF Done' in self.lines[i]:
                if region_state == 'svp':
                    qm_data['svp_electronic_energy'] = self._extract_electronic_energy(i)
                elif region_state == 'tzvp':
                    qm_data['tzvp_electronic_energy'] = self._extract_electronic_energy(i)

            if 'Dipole moment (field-independent basis, Debye)' in self.lines[i]:
                if region_state == 'svp':
                    qm_data['svp_dipole_moment'] = self._extract_dipole_moment(i + 1)
                elif region_state == 'tzvp':
                    qm_data['tzvp_dipole_moment'] = self._extract_dipole_moment(i + 1)

            if 'Isotropic polarizability' in self.lines[i]:
                qm_data['polarisability'] = self._extract_polarisability(i)

            if 'Frequencies -- ' in self.lines[i]:
                if 'frequencies' not in qm_data:
                    qm_data['frequencies'] = self._extract_frequency(i)
                else:
                    qm_data['frequencies'].extend(self._extract_frequency(i))

            if 'Zero-point correction=' in self.lines[i]:
                qm_data['zpe_correction'] = self._extract_zpe_correction(i)

            if 'Sum of electronic and thermal Enthalpies=' in self.lines[i]:
                qm_data['enthalpy_energy'] = self._extract_enthalpy_energy(i)

            if 'Sum of electronic and thermal Free Energies=' in self.lines[i]:
                qm_data['gibbs_energy'] = self._extract_gibbs_energy(i)
                qm_data['heat_capacity'] = self._extract_heat_capacity(i + 4)
                qm_data['entropy'] = self._extract_entropy(i + 4)

            if 'Alpha  occ. eigenvalues' in self.lines[i]:
                if region_state == 'svp':
                    if 'svp_occupied_orbital_energies' not in qm_data.keys():
                        qm_data['svp_occupied_orbital_energies'] = self._extract_orbital_energies(i)
                    else:
                        qm_data['svp_occupied_orbital_energies'].extend(self._extract_orbital_energies(i))
                elif region_state == 'tzvp':
                    if 'tzvp_occupied_orbital_energies' not in qm_data.keys():
                        qm_data['tzvp_occupied_orbital_energies'] = self._extract_orbital_energies(i)
                    else:
                        qm_data['tzvp_occupied_orbital_energies'].extend(self._extract_orbital_energies(i))

            if 'Alpha virt. eigenvalues' in self.lines[i]:
                if region_state == 'svp':
                    if 'svp_virtual_orbital_energies' not in qm_data.keys():
                        qm_data['svp_virtual_orbital_energies'] = self._extract_orbital_energies(i)
                    else:
                        qm_data['svp_virtual_orbital_energies'].extend(self._extract_orbital_energies(i))
                elif region_state == 'tzvp':
                    if 'tzvp_virtual_orbital_energies' not in qm_data.keys():
                        qm_data['tzvp_virtual_orbital_energies'] = self._extract_orbital_energies(i)
                    else:
                        qm_data['tzvp_virtual_orbital_energies'].extend(self._extract_orbital_energies(i))

        qm_data['nbo_data'] = self._merge_nbo_data(nbo_entries, nbo_energies)

        return qm_data

    # - - - extraction functions - - - #

    # Some of the following extraction functions are redundant in the sense that for some properties
    # the extraction procedures are identical. The distinction between these functions is kept
    # nonetheless to ensure maintainability (e.g. when the Gaussian output format changes).

    def _extract_charge(self, start_index: int):

        line_split = self.lines[start_index].split()
        return int(line_split[2])

    def _extract_stoichiometry(self, start_index: int):

        line_split = self.lines[start_index].split()
        return line_split[1]

    def _extract_molecular_mass(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[2])

    def _extract_dispersion_energy(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[4])

    def _extract_electronic_energy(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[4])

    def _extract_dipole_moment(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[7])

    def _extract_polarisability(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[5])

    def _extract_frequency(self, start_index: int):

        line_split = self.lines[start_index].split()
        return list(map(float, line_split[2:]))

    def _extract_orbital_energies(self, start_index: int):

        # line_split = self.lines[start_index].split()
        line_split = re.findall('-{0,1}[0-9]{1,}.[0-9]{1,}', self.lines[start_index])

        # build output list
        return [float(entry) for entry in line_split]

    def _extract_enthalpy_energy(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[6])

    def _extract_gibbs_energy(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[7])

    def _extract_zpe_correction(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[2])

    def _extract_heat_capacity(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[2])

    def _extract_entropy(self, start_index: int):

        line_split = self.lines[start_index].split()
        return float(line_split[3])

    def _extract_atomic_numbers(self, start_index: int):

        atomic_numbers = [int(self.lines[i].split()[1]) for i in range(start_index, start_index + self.n_atoms, 1)]

        return atomic_numbers

    def _extract_geometric_data(self, start_index: int):

        geometric_data = []

        for i in range(start_index, start_index + self.n_atoms, 1):
            # split line at any white space
            line_split = self.lines[i].split()
            # read out data (index number based on Gaussian output format)
            xyz = [float(line_split[3]), float(line_split[4]), float(line_split[5])]
            geometric_data.append(xyz)

        # also return index i to jump ahead in the file
        return geometric_data, i

    def _extract_natural_atomic_charges(self, start_index):

        natural_atomic_charges = [float(self.lines[i].split()[2]) for i in range(start_index, start_index + self.n_atoms, 1)]
        natural_electron_population = [[float(self.lines[i].split()[j]) for j in range(3, 6)] for i in range(start_index, start_index + self.n_atoms, 1)]

        # also return index i to jump ahead in the file
        return natural_atomic_charges, natural_electron_population, start_index + self.n_atoms

    def _extract_natural_electron_configuration(self, start_index: int):

        natural_electron_configuration = []

        for i in range(start_index, start_index + self.n_atoms, 1):

            # single atom electron configuration ([s, p, d, f])
            electron_configuration = [0.0, 0.0, 0.0, 0.0]

            # split line at any white space
            line_split = self.lines[i].split()
            # remove first two columns of data, rejoin and remove '[core]'
            line_cleaned = ''.join(line_split[2:]).replace('[core]', '')
            # split at '(' and ')' so that orbital type and config can be extracted
            line_cleanedSplit = re.split(r'\(|\)', line_cleaned)

            for j in range(0, len(line_cleanedSplit), 2):

                # add value to appropriate list element
                if 's' in line_cleanedSplit[j]:
                    electron_configuration[0] += float(line_cleanedSplit[j + 1])
                elif 'p' in line_cleanedSplit[j]:
                    electron_configuration[1] += float(line_cleanedSplit[j + 1])
                elif 'd' in line_cleanedSplit[j]:
                    electron_configuration[2] += float(line_cleanedSplit[j + 1])
                elif 'f' in line_cleanedSplit[j]:
                    electron_configuration[3] += float(line_cleanedSplit[j + 1])
                else:
                    continue

            # append to full list
            natural_electron_configuration.append(electron_configuration)

        # also return index i to jump ahead in the file
        return natural_electron_configuration, i

    def _extract_index_matrix(self, start_index: int):

        # setup n_atoms x n_atoms matrix for Wiberg indices
        wiberg_index_matrix = [[0 for x in range(self.n_atoms)] for y in range(self.n_atoms)]

        # counter for keeping track how many columns have been taken care of
        # this is necessary because the Gaussian output file prints the Wiberg
        # index matrix in blocks of columns
        n_columns_processed = 0

        # run until all columns have been processed
        while n_columns_processed < self.n_atoms:

            n_columns = None
            for i in range(start_index, start_index + self.n_atoms, 1):
                # split line at any white space
                line_split = self.lines[i].split()
                # drop first two columns so that only Wiberg indices remain
                line_split = line_split[2:]

                # check that the number of columns is the same
                if n_columns is None:
                    n_columns = len(line_split)
                else:
                    assert n_columns == len(line_split)

                # read out data (index number based on Gaussian output format)
                for j in range(len(line_split)):
                    # write matrix element
                    wiberg_index_matrix[i - start_index][j + n_columns_processed] = float(line_split[j])

            n_columns_processed += n_columns

            # set start_index to the next block
            start_index += self.n_atoms + 3

        # also return index i to jump ahead in the file
        return wiberg_index_matrix, i

    def _extract_lmo_bond_data(self, start_index: int):

        # output matrix
        lmo_bond_data_matrix = [[0 for x in range(self.n_atoms)] for y in range(self.n_atoms)]

        # rename for brevity
        i = start_index

        while(self.lines[i] != ''):

            line_split = self.lines[i].split()

            # get atom indices and the corresponding LMO bond order
            index_a = int(line_split[0]) - 1
            index_b = int(line_split[1]) - 1
            lmo_bond_order = float(line_split[3])

            lmo_bond_data_matrix[index_a][index_b] += lmo_bond_order
            lmo_bond_data_matrix[index_b][index_a] += lmo_bond_order

            i += 1

        return lmo_bond_data_matrix, i

    def _extract_nbo_energies(self, start_index: int):

        data = []

        # rename index for brevity
        i = start_index

        while('NATURAL LOCALIZED MOLECULAR ORBITAL' not in self.lines[i]):

            line_split = list(filter(None, re.split(r'\(|\)|([0-9]+)-| ', self.lines[i])))

            if len(line_split) > 3:

                if line_split[1] == 'LP' or line_split[1] == 'LV':
                    energy = float(line_split[6])
                elif line_split[1] == 'BD' or line_split[1] == 'BD*':
                    energy = float(line_split[8])
                elif line_split[1] == '3C' or line_split[1] == '3C*' or line_split[1] == '3Cn':
                    energy = float(line_split[10])
                else:
                    i += 1
                    continue

                id = int(line_split[0].replace('.', ''))
                data.append([id, energy])

            i += 1

        # also return index i to jump ahead in the file
        return data, i

    def _extract_nbo_data(self, start_index: int):

        # final output variable
        data = []

        # rename index for brevity
        i = start_index

        while not self.lines[i] == '':

            # split line at any white space
            line_split = self.lines[i].replace('(', '').split()
            if len(line_split) > 3:

                # lone pairs
                if line_split[2] == 'LP' or line_split[2] == 'LV':
                    data.append(self._extract_lone_pair_data(i))

                # bonds
                if line_split[2] == 'BD' or line_split[2] == 'BD*':
                    data.append(self._extract_bonding_data(i))

                # 3 centers
                if line_split[2] == '3C' or line_split[2] == '3C*' or line_split[2] == '3Cn':
                    data.append(self._extract_three_center_data(i))

            i += 1

        # also return index i to jump ahead in the file
        return data, i

    def _get_nbo_occupations(self, text: str):

        result = re.findall(r'[^\s]\((.{4,6})%\)', text)
        occupations = list(map(float, result))

        # check that length of occupation list is correct
        # append zeros if needed
        while len(occupations) < 4:
            occupations.append(0.0)

        return occupations

    def _get_nbo_contributions(self, text: str):

        result = re.findall(r'\((.{4,6})%\)', text)
        return float(result[0]) / 100

    def _extract_lone_pair_data(self, start_index: int):

        # get ID of entry
        id = int(self.lines[start_index].split('.')[0])

        line_split = list(filter(None, re.split(r'\(|\)| ', self.lines[start_index])))

        # obtain atom position
        atom_position = [int(line_split[5]) - 1]
        # obtain occupation
        full_occupation = float(line_split[1])
        # nbo type
        nbo_type = line_split[2]

        # get occupation from both lines using regex (values in brackets)
        merged_lines = (self.lines[start_index] + self.lines[start_index + 1]).replace(' ', '')
        occupations = self._get_nbo_occupations(merged_lines)

        # return id, atom position, contribution, occupation and percent occupations
        # divide occupations by 100 (get rid of %)
        # for one atom NBOs the contribution is simply a list with one entry 1.
        return [id, nbo_type, atom_position, [1], full_occupation, [x / 100 for x in occupations]]

    def _extract_bonding_data(self, start_index: int):

        # get ID of entry
        id = int(self.lines[start_index].split('.')[0])

        line_split = list(filter(None, re.split(r'\(|\)|-| ', self.lines[start_index])))

        # obtain atom positions
        atom_positions = [int(line_split[-3]) - 1, int(line_split[-1]) - 1]
        # obtain occupation
        full_occupation = float(line_split[1])
        # nbo type
        nbo_type = line_split[2]

        contributions = []
        occupations = [0.0, 0.0, 0.0, 0.0]

        # line counter
        i = start_index + 1
        for k in range(2):

            while '(' not in self.lines[i]:
                i += 1

            # get occupation from both lines using regex (values in brackets)
            merged_lines = (self.lines[i] + self.lines[i + 1]).replace(' ', '')
            # append corresponding contribution
            contributions.append(self._get_nbo_contributions(merged_lines))
            # add occupations
            occupations = list(map(add, occupations, self._get_nbo_occupations(merged_lines)))

            # increment counter
            i += 2

        # check that length of occupation list is correct
        assert len(occupations) == 4

        # return id, atom position, occupation and percent occupations
        # divide occupations by 100 (get rid of %)
        return [id, nbo_type, atom_positions, contributions, full_occupation, [x / 200 for x in occupations]]
        # return atom_positions, [x / 200 for x in occupations]

    def _extract_three_center_data(self, start_index: int):

        # get ID of entry
        id = int(self.lines[start_index].split('.')[0])

        line_split = list(filter(None, re.split(r'\(|\)|-| ', self.lines[start_index])))

        # obtain atom positions
        atom_positions = [int(line_split[-5]) - 1, int(line_split[-3]) - 1, int(line_split[-1]) - 1]
        # obtain occupation
        full_occupation = float(line_split[1])
        # nbo type
        nbo_type = line_split[2]

        contributions = []
        occupations = [0.0, 0.0, 0.0, 0.0]

        # line counter
        i = start_index + 1
        for k in range(3):

            while '(' not in self.lines[i]:
                i += 1

            # get occupation from both lines using regex (values in brackets)
            merged_lines = (self.lines[i] + self.lines[i + 1]).replace(' ', '')
            # append corresponding contribution
            contributions.append(self._get_nbo_contributions(merged_lines))
            # add occupations
            occupations = list(map(add, occupations, self._get_nbo_occupations(merged_lines)))

            # increment counter
            i += 2

        # check that length of occupation list is correct
        assert len(occupations) == 4

        # return id, atom position, occupation and percent occupations
        # divide occupations by 100 (get rid of %)
        return [id, nbo_type, atom_positions, contributions, full_occupation, [x / 300 for x in occupations]]

    def _extract_sopa_data(self, start_index: int):

        # rename index for brevity
        i = start_index

        # return variable
        sopa_data = []

        while 'NATURAL BOND ORBITALS' not in self.lines[i]:

            # line_split = self.lines[i]
            line_split = self.lines[i].split()
            if len(line_split) > 6:

                nbo_ids = list(map(int, re.findall(r'([0-9]{1,5})\. [A-Z,0-9]{2}', self.lines[i])))
                energies = list(map(float, line_split[-3:]))

                sopa_data.append([nbo_ids, energies])

            i += 1

        return sopa_data, i

    def _extract_hyper_node_data(self, start_index: int):

        # rename index for brevity
        i = start_index

        # return variable
        three_center_nbos = []

        while self.lines[i] != '':

            line_split = self.lines[i].split()
            three_center_nbos.append([[int(line_split[9]), int(line_split[10])], float(line_split[8])])

            i += 1

        return three_center_nbos

    def _merge_nbo_data(self, nbo_entries: list, nbo_energies: list) -> list[list]:

        """Merges NBO entries (energies and occupations) and stores them as a list of lists.

        Returns:
            list[list]: The list of NBO entries
        """

        nbo_data = []

        # readout IDs of energies to match energies to corresponding nbo entries
        energy_ids = [x[0] for x in nbo_energies]
        for i in range(len(nbo_entries)):

            # get the index of the ID of the current nbo data point
            nbo_energy_index = energy_ids.index(nbo_entries[i][0])
            nbo_energy = nbo_energies[nbo_energy_index][1]

            # # list assignment of NBO data
            # 0: ID
            # 1: NBO type
            # 2: atom indices
            # 3: energy
            # 4: contributions
            # 5: occupation
            # 6: orbital occupations
            nbo_data_point = [
                nbo_entries[i][0],
                nbo_entries[i][1],
                nbo_entries[i][2],
                nbo_energy,
                nbo_entries[i][3],
                nbo_entries[i][4],
                nbo_entries[i][5],
            ]

            nbo_data.append(nbo_data_point)

        return nbo_data
