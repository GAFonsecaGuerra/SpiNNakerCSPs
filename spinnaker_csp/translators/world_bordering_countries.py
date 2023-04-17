# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
# -----------------------------------------------------------------------------------------------------------------------
"""Define a CSP representation of the map of the world.

Countries are represented by numbers acording to the list in alphabetical order as seen in the world_countries
dictionary. Constraints corresponding to bordering countries of the world as defined by the United Nations as shown in
the world_borders dictionary.

The countries and bordering correspond to data from the United Nations available in Mathematica Wolfram
(Wolfram Research, 2017).
"""
world_countries = [
    {"Afghanistan": 0},
    {"Albania": 1},
    {"Algeria": 2},
    {"Andorra": 3},
    {"Angola": 4},
    {"Antigua and Barbuda": 5},
    {"Argentina": 6},
    {"Armenia": 7},
    {"Australia": 8},
    {"Austria": 9},
    {"Azerbaijan": 10},
    {"Bahamas": 11},
    {"Bahrain": 12},
    {"Bangladesh": 13},
    {"Barbados": 14},
    {"Belarus": 15},
    {"Belgium": 16},
    {"Belize": 17},
    {"Benin": 18},
    {"Bhutan": 19},
    {"Bolivia": 20},
    {"Bosnia and Herzegovina": 21},
    {"Botswana": 22},
    {"Brazil": 23},
    {"Brunei": 24},
    {"Bulgaria": 25},
    {"Burkina Faso": 26},
    {"Burundi": 27},
    {"Cambodia": 28},
    {"Cameroon": 29},
    {"Canada": 30},
    {"Cape Verde": 31},
    {"Central African Republic": 32},
    {"Chad": 33},
    {"Chile": 34},
    {"China": 35},
    {"Colombia": 36},
    {"Comoros": 37},
    {"Costa Rica": 38},
    {"Croatia": 39},
    {"Cuba": 40},
    {"Cyprus": 41},
    {"Czech Republic": 42},
    {"Democratic Republic of the Congo": 43},
    {"Denmark": 44},
    {"Djibouti": 45},
    {"Dominica": 46},
    {"Dominican Republic": 47},
    {"East Timor": 48},
    {"Ecuador": 49},
    {"Egypt": 50},
    {"El Salvador": 51},
    {"Equatorial Guinea": 52},
    {"Eritrea": 53},
    {"Estonia": 54},
    {"Ethiopia": 55},
    {"Fiji": 56},
    {"Finland": 57},
    {"France": 58},
    {"Gabon": 59},
    {"Gambia": 60},
    {"Georgia": 61},
    {"Germany": 62},
    {"Ghana": 63},
    {"Greece": 64},
    {"Grenada": 65},
    {"Guatemala": 66},
    {"Guinea": 67},
    {"Guinea-Bissau": 68},
    {"Guyana": 69},
    {"Haiti": 70},
    {"Honduras": 71},
    {"Hungary": 72},
    {"Iceland": 73},
    {"India": 74},
    {"Indonesia": 75},
    {"Iran": 76},
    {"Iraq": 77},
    {"Ireland": 78},
    {"Israel": 79},
    {"Italy": 80},
    {"Ivory Coast": 81},
    {"Jamaica": 82},
    {"Japan": 83},
    {"Jordan": 84},
    {"Kazakhstan": 85},
    {"Kenya": 86},
    {"Kiribati": 87},
    {"Kuwait": 88},
    {"Kyrgyzstan": 89},
    {"Laos": 90},
    {"Latvia": 91},
    {"Lebanon": 92},
    {"Lesotho": 93},
    {"Liberia": 94},
    {"Libya": 95},
    {"Liechtenstein": 96},
    {"Lithuania": 97},
    {"Luxembourg": 98},
    {"Macedonia": 99},
    {"Madagascar": 100},
    {"Malawi": 101},
    {"Malaysia": 102},
    {"Maldives": 103},
    {"Mali": 104},
    {"Malta": 105},
    {"Marshall Islands": 106},
    {"Mauritania": 107},
    {"Mauritius": 108},
    {"Mexico": 109},
    {"Micronesia": 110},
    {"Moldova": 111},
    {"Monaco": 112},
    {"Mongolia": 113},
    {"Montenegro": 114},
    {"Morocco": 115},
    {"Mozambique": 116},
    {"Myanmar": 117},
    {"Namibia": 118},
    {"Nauru": 119},
    {"Nepal": 120},
    {"Netherlands": 121},
    {"New Zealand": 122},
    {"Nicaragua": 123},
    {"Niger": 124},
    {"Nigeria": 125},
    {"North Korea": 126},
    {"Norway": 127},
    {"Oman": 128},
    {"Pakistan": 129},
    {"Palau": 130},
    {"Panama": 131},
    {"Papua New Guinea": 132},
    {"Paraguay": 133},
    {"Peru": 134},
    {"Philippines": 135},
    {"Poland": 136},
    {"Portugal": 137},
    {"Qatar": 138},
    {"Republic of the Congo": 139},
    {"Romania": 140},
    {"Russia": 141},
    {"Rwanda": 142},
    {"Saint Kitts and Nevis": 143},
    {"Saint Lucia": 144},
    {"Saint Vincent and the Grenadines": 145},
    {"Samoa": 146},
    {"San Marino": 147},
    {"Sao Tome and Principe": 148},
    {"Saudi Arabia": 149},
    {"Senegal": 150},
    {"Serbia": 151},
    {"Seychelles": 152},
    {"Sierra Leone": 153},
    {"Singapore": 154},
    {"Slovakia": 155},
    {"Slovenia": 156},
    {"Solomon Islands": 157},
    {"Somalia": 158},
    {"South Africa": 159},
    {"South Korea": 160},
    {"South Sudan": 161},
    {"Spain": 162},
    {"Sri Lanka": 163},
    {"Sudan": 164},
    {"Suriname": 165},
    {"Swaziland": 166},
    {"Sweden": 167},
    {"Switzerland": 168},
    {"Syria": 169},
    {"Tajikistan": 170},
    {"Tanzania": 171},
    {"Thailand": 172},
    {"Togo": 173},
    {"Tonga": 174},
    {"Trinidad and Tobago": 175},
    {"Tunisia": 176},
    {"Turkey": 177},
    {"Turkmenistan": 178},
    {"Tuvalu": 179},
    {"Uganda": 180},
    {"Ukraine": 181},
    {"United Arab Emirates": 182},
    {"United Kingdom": 183},
    {"United States": 184},
    {"Uruguay": 185},
    {"Uzbekistan": 186},
    {"Vanuatu": 187},
    {"Venezuela": 188},
    {"Vietnam": 189},
    {"Yemen": 190},
    {"Zambia": 191},
    {"Zimbabwe": 192},
]

world_borders = [
    {"source": 0, "target": 35},
    {"source": 0, "target": 76},
    {"source": 0, "target": 129},
    {"source": 0, "target": 170},
    {"source": 0, "target": 178},
    {"source": 0, "target": 186},
    {"source": 1, "target": 64},
    {"source": 1, "target": 99},
    {"source": 1, "target": 114},
    {"source": 2, "target": 95},
    {"source": 2, "target": 104},
    {"source": 2, "target": 107},
    {"source": 2, "target": 115},
    {"source": 2, "target": 124},
    {"source": 2, "target": 176},
    {"source": 3, "target": 58},
    {"source": 3, "target": 162},
    {"source": 4, "target": 43},
    {"source": 4, "target": 118},
    {"source": 4, "target": 139},
    {"source": 4, "target": 191},
    {"source": 6, "target": 20},
    {"source": 6, "target": 23},
    {"source": 6, "target": 34},
    {"source": 6, "target": 133},
    {"source": 6, "target": 185},
    {"source": 7, "target": 10},
    {"source": 7, "target": 61},
    {"source": 7, "target": 76},
    {"source": 7, "target": 177},
    {"source": 9, "target": 42},
    {"source": 9, "target": 62},
    {"source": 9, "target": 72},
    {"source": 9, "target": 80},
    {"source": 9, "target": 96},
    {"source": 9, "target": 155},
    {"source": 9, "target": 156},
    {"source": 9, "target": 168},
    {"source": 10, "target": 7},
    {"source": 10, "target": 61},
    {"source": 10, "target": 76},
    {"source": 10, "target": 141},
    {"source": 10, "target": 177},
    {"source": 13, "target": 74},
    {"source": 13, "target": 117},
    {"source": 15, "target": 91},
    {"source": 15, "target": 97},
    {"source": 15, "target": 136},
    {"source": 15, "target": 141},
    {"source": 15, "target": 181},
    {"source": 16, "target": 58},
    {"source": 16, "target": 62},
    {"source": 16, "target": 98},
    {"source": 16, "target": 121},
    {"source": 17, "target": 66},
    {"source": 17, "target": 109},
    {"source": 18, "target": 26},
    {"source": 18, "target": 124},
    {"source": 18, "target": 125},
    {"source": 18, "target": 173},
    {"source": 19, "target": 35},
    {"source": 19, "target": 74},
    {"source": 20, "target": 6},
    {"source": 20, "target": 23},
    {"source": 20, "target": 34},
    {"source": 20, "target": 133},
    {"source": 20, "target": 134},
    {"source": 21, "target": 39},
    {"source": 21, "target": 114},
    {"source": 21, "target": 151},
    {"source": 22, "target": 118},
    {"source": 22, "target": 159},
    {"source": 22, "target": 191},
    {"source": 22, "target": 192},
    {"source": 23, "target": 6},
    {"source": 23, "target": 20},
    {"source": 23, "target": 36},
    {"source": 23, "target": 69},
    {"source": 23, "target": 133},
    {"source": 23, "target": 134},
    {"source": 23, "target": 165},
    {"source": 23, "target": 185},
    {"source": 23, "target": 188},
    {"source": 24, "target": 102},
    {"source": 25, "target": 64},
    {"source": 25, "target": 99},
    {"source": 25, "target": 140},
    {"source": 25, "target": 151},
    {"source": 25, "target": 177},
    {"source": 26, "target": 18},
    {"source": 26, "target": 63},
    {"source": 26, "target": 81},
    {"source": 26, "target": 104},
    {"source": 26, "target": 124},
    {"source": 26, "target": 173},
    {"source": 27, "target": 43},
    {"source": 27, "target": 142},
    {"source": 27, "target": 171},
    {"source": 28, "target": 90},
    {"source": 28, "target": 172},
    {"source": 28, "target": 189},
    {"source": 29, "target": 32},
    {"source": 29, "target": 33},
    {"source": 29, "target": 52},
    {"source": 29, "target": 59},
    {"source": 29, "target": 125},
    {"source": 29, "target": 139},
    {"source": 30, "target": 184},
    {"source": 32, "target": 29},
    {"source": 32, "target": 33},
    {"source": 32, "target": 43},
    {"source": 32, "target": 139},
    {"source": 32, "target": 161},
    {"source": 32, "target": 164},
    {"source": 33, "target": 29},
    {"source": 33, "target": 32},
    {"source": 33, "target": 95},
    {"source": 33, "target": 124},
    {"source": 33, "target": 125},
    {"source": 33, "target": 164},
    {"source": 34, "target": 6},
    {"source": 34, "target": 20},
    {"source": 34, "target": 134},
    {"source": 35, "target": 0},
    {"source": 35, "target": 19},
    {"source": 35, "target": 74},
    {"source": 35, "target": 85},
    {"source": 35, "target": 89},
    {"source": 35, "target": 90},
    {"source": 35, "target": 113},
    {"source": 35, "target": 117},
    {"source": 35, "target": 120},
    {"source": 35, "target": 126},
    {"source": 35, "target": 129},
    {"source": 35, "target": 141},
    {"source": 35, "target": 170},
    {"source": 35, "target": 189},
    {"source": 36, "target": 23},
    {"source": 36, "target": 49},
    {"source": 36, "target": 131},
    {"source": 36, "target": 134},
    {"source": 36, "target": 188},
    {"source": 38, "target": 123},
    {"source": 38, "target": 131},
    {"source": 39, "target": 21},
    {"source": 39, "target": 72},
    {"source": 39, "target": 114},
    {"source": 39, "target": 151},
    {"source": 39, "target": 156},
    {"source": 42, "target": 9},
    {"source": 42, "target": 62},
    {"source": 42, "target": 136},
    {"source": 42, "target": 155},
    {"source": 43, "target": 4},
    {"source": 43, "target": 27},
    {"source": 43, "target": 32},
    {"source": 43, "target": 139},
    {"source": 43, "target": 142},
    {"source": 43, "target": 161},
    {"source": 43, "target": 171},
    {"source": 43, "target": 180},
    {"source": 43, "target": 191},
    {"source": 44, "target": 62},
    {"source": 45, "target": 53},
    {"source": 45, "target": 55},
    {"source": 45, "target": 158},
    {"source": 47, "target": 70},
    {"source": 48, "target": 75},
    {"source": 49, "target": 36},
    {"source": 49, "target": 134},
    {"source": 50, "target": 79},
    {"source": 50, "target": 95},
    {"source": 50, "target": 164},
    {"source": 51, "target": 66},
    {"source": 51, "target": 71},
    {"source": 52, "target": 29},
    {"source": 52, "target": 59},
    {"source": 53, "target": 45},
    {"source": 53, "target": 55},
    {"source": 53, "target": 164},
    {"source": 54, "target": 91},
    {"source": 54, "target": 141},
    {"source": 55, "target": 45},
    {"source": 55, "target": 53},
    {"source": 55, "target": 86},
    {"source": 55, "target": 158},
    {"source": 55, "target": 161},
    {"source": 55, "target": 164},
    {"source": 57, "target": 127},
    {"source": 57, "target": 141},
    {"source": 57, "target": 167},
    {"source": 58, "target": 3},
    {"source": 58, "target": 16},
    {"source": 58, "target": 62},
    {"source": 58, "target": 80},
    {"source": 58, "target": 98},
    {"source": 58, "target": 112},
    {"source": 58, "target": 162},
    {"source": 58, "target": 168},
    {"source": 59, "target": 29},
    {"source": 59, "target": 52},
    {"source": 59, "target": 139},
    {"source": 60, "target": 150},
    {"source": 61, "target": 7},
    {"source": 61, "target": 10},
    {"source": 61, "target": 141},
    {"source": 61, "target": 177},
    {"source": 62, "target": 9},
    {"source": 62, "target": 16},
    {"source": 62, "target": 42},
    {"source": 62, "target": 44},
    {"source": 62, "target": 58},
    {"source": 62, "target": 98},
    {"source": 62, "target": 121},
    {"source": 62, "target": 136},
    {"source": 62, "target": 168},
    {"source": 63, "target": 26},
    {"source": 63, "target": 81},
    {"source": 63, "target": 173},
    {"source": 64, "target": 1},
    {"source": 64, "target": 25},
    {"source": 64, "target": 99},
    {"source": 64, "target": 177},
    {"source": 66, "target": 17},
    {"source": 66, "target": 51},
    {"source": 66, "target": 71},
    {"source": 66, "target": 109},
    {"source": 67, "target": 68},
    {"source": 67, "target": 81},
    {"source": 67, "target": 94},
    {"source": 67, "target": 104},
    {"source": 67, "target": 150},
    {"source": 67, "target": 153},
    {"source": 68, "target": 67},
    {"source": 68, "target": 150},
    {"source": 69, "target": 23},
    {"source": 69, "target": 165},
    {"source": 69, "target": 188},
    {"source": 70, "target": 47},
    {"source": 71, "target": 51},
    {"source": 71, "target": 66},
    {"source": 71, "target": 123},
    {"source": 72, "target": 9},
    {"source": 72, "target": 39},
    {"source": 72, "target": 140},
    {"source": 72, "target": 151},
    {"source": 72, "target": 155},
    {"source": 72, "target": 156},
    {"source": 72, "target": 181},
    {"source": 74, "target": 13},
    {"source": 74, "target": 19},
    {"source": 74, "target": 35},
    {"source": 74, "target": 117},
    {"source": 74, "target": 120},
    {"source": 74, "target": 129},
    {"source": 75, "target": 48},
    {"source": 75, "target": 102},
    {"source": 75, "target": 132},
    {"source": 76, "target": 0},
    {"source": 76, "target": 7},
    {"source": 76, "target": 10},
    {"source": 76, "target": 77},
    {"source": 76, "target": 129},
    {"source": 76, "target": 177},
    {"source": 76, "target": 178},
    {"source": 77, "target": 76},
    {"source": 77, "target": 84},
    {"source": 77, "target": 88},
    {"source": 77, "target": 149},
    {"source": 77, "target": 169},
    {"source": 77, "target": 177},
    {"source": 78, "target": 183},
    {"source": 79, "target": 50},
    {"source": 79, "target": 84},
    {"source": 79, "target": 92},
    {"source": 79, "target": 169},
    {"source": 80, "target": 9},
    {"source": 80, "target": 58},
    {"source": 80, "target": 147},
    {"source": 80, "target": 156},
    {"source": 80, "target": 168},
    {"source": 81, "target": 26},
    {"source": 81, "target": 63},
    {"source": 81, "target": 67},
    {"source": 81, "target": 94},
    {"source": 81, "target": 104},
    {"source": 84, "target": 77},
    {"source": 84, "target": 79},
    {"source": 84, "target": 149},
    {"source": 84, "target": 169},
    {"source": 85, "target": 35},
    {"source": 85, "target": 89},
    {"source": 85, "target": 141},
    {"source": 85, "target": 178},
    {"source": 85, "target": 186},
    {"source": 86, "target": 55},
    {"source": 86, "target": 158},
    {"source": 86, "target": 161},
    {"source": 86, "target": 171},
    {"source": 86, "target": 180},
    {"source": 88, "target": 77},
    {"source": 88, "target": 149},
    {"source": 89, "target": 35},
    {"source": 89, "target": 85},
    {"source": 89, "target": 170},
    {"source": 89, "target": 186},
    {"source": 90, "target": 28},
    {"source": 90, "target": 35},
    {"source": 90, "target": 117},
    {"source": 90, "target": 172},
    {"source": 90, "target": 189},
    {"source": 91, "target": 15},
    {"source": 91, "target": 54},
    {"source": 91, "target": 97},
    {"source": 91, "target": 141},
    {"source": 92, "target": 79},
    {"source": 92, "target": 169},
    {"source": 93, "target": 159},
    {"source": 94, "target": 67},
    {"source": 94, "target": 81},
    {"source": 94, "target": 153},
    {"source": 95, "target": 2},
    {"source": 95, "target": 33},
    {"source": 95, "target": 50},
    {"source": 95, "target": 124},
    {"source": 95, "target": 164},
    {"source": 95, "target": 176},
    {"source": 96, "target": 9},
    {"source": 96, "target": 168},
    {"source": 97, "target": 15},
    {"source": 97, "target": 91},
    {"source": 97, "target": 136},
    {"source": 97, "target": 141},
    {"source": 98, "target": 16},
    {"source": 98, "target": 58},
    {"source": 98, "target": 62},
    {"source": 99, "target": 1},
    {"source": 99, "target": 25},
    {"source": 99, "target": 64},
    {"source": 99, "target": 151},
    {"source": 101, "target": 116},
    {"source": 101, "target": 171},
    {"source": 101, "target": 191},
    {"source": 102, "target": 24},
    {"source": 102, "target": 75},
    {"source": 102, "target": 172},
    {"source": 104, "target": 2},
    {"source": 104, "target": 26},
    {"source": 104, "target": 67},
    {"source": 104, "target": 81},
    {"source": 104, "target": 107},
    {"source": 104, "target": 124},
    {"source": 104, "target": 150},
    {"source": 107, "target": 2},
    {"source": 107, "target": 104},
    {"source": 107, "target": 150},
    {"source": 109, "target": 17},
    {"source": 109, "target": 66},
    {"source": 109, "target": 184},
    {"source": 111, "target": 140},
    {"source": 111, "target": 181},
    {"source": 112, "target": 58},
    {"source": 113, "target": 35},
    {"source": 113, "target": 141},
    {"source": 114, "target": 1},
    {"source": 114, "target": 21},
    {"source": 114, "target": 39},
    {"source": 114, "target": 151},
    {"source": 115, "target": 2},
    {"source": 115, "target": 162},
    {"source": 116, "target": 101},
    {"source": 116, "target": 159},
    {"source": 116, "target": 166},
    {"source": 116, "target": 171},
    {"source": 116, "target": 191},
    {"source": 116, "target": 192},
    {"source": 117, "target": 13},
    {"source": 117, "target": 35},
    {"source": 117, "target": 74},
    {"source": 117, "target": 90},
    {"source": 117, "target": 172},
    {"source": 118, "target": 4},
    {"source": 118, "target": 22},
    {"source": 118, "target": 159},
    {"source": 118, "target": 191},
    {"source": 120, "target": 35},
    {"source": 120, "target": 74},
    {"source": 121, "target": 16},
    {"source": 121, "target": 62},
    {"source": 123, "target": 38},
    {"source": 123, "target": 71},
    {"source": 124, "target": 2},
    {"source": 124, "target": 18},
    {"source": 124, "target": 26},
    {"source": 124, "target": 33},
    {"source": 124, "target": 95},
    {"source": 124, "target": 104},
    {"source": 124, "target": 125},
    {"source": 125, "target": 18},
    {"source": 125, "target": 29},
    {"source": 125, "target": 33},
    {"source": 125, "target": 124},
    {"source": 126, "target": 35},
    {"source": 126, "target": 141},
    {"source": 126, "target": 160},
    {"source": 127, "target": 57},
    {"source": 127, "target": 141},
    {"source": 127, "target": 167},
    {"source": 128, "target": 149},
    {"source": 128, "target": 182},
    {"source": 128, "target": 190},
    {"source": 129, "target": 0},
    {"source": 129, "target": 35},
    {"source": 129, "target": 74},
    {"source": 129, "target": 76},
    {"source": 131, "target": 36},
    {"source": 131, "target": 38},
    {"source": 132, "target": 75},
    {"source": 133, "target": 6},
    {"source": 133, "target": 20},
    {"source": 133, "target": 23},
    {"source": 134, "target": 20},
    {"source": 134, "target": 23},
    {"source": 134, "target": 34},
    {"source": 134, "target": 36},
    {"source": 134, "target": 49},
    {"source": 136, "target": 15},
    {"source": 136, "target": 42},
    {"source": 136, "target": 62},
    {"source": 136, "target": 97},
    {"source": 136, "target": 141},
    {"source": 136, "target": 155},
    {"source": 136, "target": 181},
    {"source": 137, "target": 162},
    {"source": 138, "target": 149},
    {"source": 139, "target": 4},
    {"source": 139, "target": 29},
    {"source": 139, "target": 32},
    {"source": 139, "target": 43},
    {"source": 139, "target": 59},
    {"source": 140, "target": 25},
    {"source": 140, "target": 72},
    {"source": 140, "target": 111},
    {"source": 140, "target": 151},
    {"source": 140, "target": 181},
    {"source": 141, "target": 10},
    {"source": 141, "target": 15},
    {"source": 141, "target": 35},
    {"source": 141, "target": 54},
    {"source": 141, "target": 57},
    {"source": 141, "target": 61},
    {"source": 141, "target": 85},
    {"source": 141, "target": 91},
    {"source": 141, "target": 97},
    {"source": 141, "target": 113},
    {"source": 141, "target": 126},
    {"source": 141, "target": 127},
    {"source": 141, "target": 136},
    {"source": 141, "target": 181},
    {"source": 142, "target": 27},
    {"source": 142, "target": 43},
    {"source": 142, "target": 171},
    {"source": 142, "target": 180},
    {"source": 147, "target": 80},
    {"source": 149, "target": 77},
    {"source": 149, "target": 84},
    {"source": 149, "target": 88},
    {"source": 149, "target": 128},
    {"source": 149, "target": 138},
    {"source": 149, "target": 182},
    {"source": 149, "target": 190},
    {"source": 150, "target": 60},
    {"source": 150, "target": 67},
    {"source": 150, "target": 68},
    {"source": 150, "target": 104},
    {"source": 150, "target": 107},
    {"source": 151, "target": 21},
    {"source": 151, "target": 25},
    {"source": 151, "target": 39},
    {"source": 151, "target": 72},
    {"source": 151, "target": 99},
    {"source": 151, "target": 114},
    {"source": 151, "target": 140},
    {"source": 153, "target": 67},
    {"source": 153, "target": 94},
    {"source": 155, "target": 9},
    {"source": 155, "target": 42},
    {"source": 155, "target": 72},
    {"source": 155, "target": 136},
    {"source": 155, "target": 181},
    {"source": 156, "target": 9},
    {"source": 156, "target": 39},
    {"source": 156, "target": 72},
    {"source": 156, "target": 80},
    {"source": 158, "target": 45},
    {"source": 158, "target": 55},
    {"source": 158, "target": 86},
    {"source": 159, "target": 22},
    {"source": 159, "target": 93},
    {"source": 159, "target": 116},
    {"source": 159, "target": 118},
    {"source": 159, "target": 166},
    {"source": 159, "target": 192},
    {"source": 160, "target": 126},
    {"source": 161, "target": 32},
    {"source": 161, "target": 43},
    {"source": 161, "target": 55},
    {"source": 161, "target": 86},
    {"source": 161, "target": 164},
    {"source": 161, "target": 180},
    {"source": 162, "target": 3},
    {"source": 162, "target": 58},
    {"source": 162, "target": 115},
    {"source": 162, "target": 137},
    {"source": 164, "target": 32},
    {"source": 164, "target": 33},
    {"source": 164, "target": 50},
    {"source": 164, "target": 53},
    {"source": 164, "target": 55},
    {"source": 164, "target": 95},
    {"source": 164, "target": 161},
    {"source": 165, "target": 23},
    {"source": 165, "target": 69},
    {"source": 166, "target": 116},
    {"source": 166, "target": 159},
    {"source": 167, "target": 57},
    {"source": 167, "target": 127},
    {"source": 168, "target": 9},
    {"source": 168, "target": 58},
    {"source": 168, "target": 62},
    {"source": 168, "target": 80},
    {"source": 168, "target": 96},
    {"source": 169, "target": 77},
    {"source": 169, "target": 79},
    {"source": 169, "target": 84},
    {"source": 169, "target": 92},
    {"source": 169, "target": 177},
    {"source": 170, "target": 0},
    {"source": 170, "target": 35},
    {"source": 170, "target": 89},
    {"source": 170, "target": 186},
    {"source": 171, "target": 27},
    {"source": 171, "target": 43},
    {"source": 171, "target": 86},
    {"source": 171, "target": 101},
    {"source": 171, "target": 116},
    {"source": 171, "target": 142},
    {"source": 171, "target": 180},
    {"source": 171, "target": 191},
    {"source": 172, "target": 28},
    {"source": 172, "target": 90},
    {"source": 172, "target": 102},
    {"source": 172, "target": 117},
    {"source": 173, "target": 18},
    {"source": 173, "target": 26},
    {"source": 173, "target": 63},
    {"source": 176, "target": 2},
    {"source": 176, "target": 95},
    {"source": 177, "target": 7},
    {"source": 177, "target": 10},
    {"source": 177, "target": 25},
    {"source": 177, "target": 61},
    {"source": 177, "target": 64},
    {"source": 177, "target": 76},
    {"source": 177, "target": 77},
    {"source": 177, "target": 169},
    {"source": 178, "target": 0},
    {"source": 178, "target": 76},
    {"source": 178, "target": 85},
    {"source": 178, "target": 186},
    {"source": 180, "target": 43},
    {"source": 180, "target": 86},
    {"source": 180, "target": 142},
    {"source": 180, "target": 161},
    {"source": 180, "target": 171},
    {"source": 181, "target": 15},
    {"source": 181, "target": 72},
    {"source": 181, "target": 111},
    {"source": 181, "target": 136},
    {"source": 181, "target": 140},
    {"source": 181, "target": 141},
    {"source": 181, "target": 155},
    {"source": 182, "target": 128},
    {"source": 182, "target": 149},
    {"source": 183, "target": 78},
    {"source": 184, "target": 30},
    {"source": 184, "target": 109},
    {"source": 185, "target": 6},
    {"source": 185, "target": 23},
    {"source": 186, "target": 0},
    {"source": 186, "target": 85},
    {"source": 186, "target": 89},
    {"source": 186, "target": 170},
    {"source": 186, "target": 178},
    {"source": 188, "target": 23},
    {"source": 188, "target": 36},
    {"source": 188, "target": 69},
    {"source": 189, "target": 28},
    {"source": 189, "target": 35},
    {"source": 189, "target": 90},
    {"source": 190, "target": 128},
    {"source": 190, "target": 149},
    {"source": 191, "target": 4},
    {"source": 191, "target": 22},
    {"source": 191, "target": 43},
    {"source": 191, "target": 101},
    {"source": 191, "target": 116},
    {"source": 191, "target": 118},
    {"source": 191, "target": 171},
    {"source": 191, "target": 192},
    {"source": 192, "target": 22},
    {"source": 192, "target": 116},
    {"source": 192, "target": 159},
    {"source": 192, "target": 191},
]
