import os
import re
import subprocess
import yaml
import pandas as pd
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from pathlib import Path
from confidence import Configuration
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import load_json, get_llh_estimate
from scipy.stats import gaussian_kde
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = os.path.join(SCRIPT_PATH, "resources")
SIN_PATTERN = re.compile(r'[A-Z]{4}[0-9]{4}NL#[0-9]{2}')


class Models(Enum):
    EUROFORMIX = 'model-euroformix'
    MIXCAL = 'model-mixcal'
    LRMIX = 'model-lrmix'

import time
start = time.time()
class DNAStatistX:

    def __init__(self, jar_file: str, frequencies_file: str, model: str, kit: str, threads: int,
                 thresholds: Configuration, method: str, output_file: Optional[str] = 'scratch/results'):
        self.jar_file = os.path.join(RESOURCES_PATH, jar_file)
        self.frequencies = os.path.join(RESOURCES_PATH, frequencies_file)
        self.model = model
        self.kit = kit
        self.threads = threads
        self.thresholds = tuple(f'{k}={v}' for k, v in thresholds.items() if k != 'DEFAULT')
        self.default_threshold = thresholds.DEFAULT
        self.method = method
        self.reference_file = Path(os.path.join(RESOURCES_PATH, 'reference_file.txt'))
        self._output_file = output_file

    def run(self, trace_file: Path, n_contributors: int) -> tuple[float, dict[str, Any]]:
        cmd = self.construct_cmd(trace_file, n_contributors)
        self.cleanup_output_files()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        marg_likelihood_per_locus_genotype = self.parse_output_to_marg_lh(process)
        if len(marg_likelihood_per_locus_genotype) == 0:
            raise ValueError(f"No marginalized likelihoods parsed. Process.stderr returns the following: "
                             f"{"\n".join([line for line in process.stderr])}")

        process.wait()
        return get_llh_estimate(load_json(f"{self.output_file}.json"), 'H2'), marg_likelihood_per_locus_genotype

    def parse_output_to_marg_lh(self, process: subprocess.Popen) -> dict:
        """
        Parse the raw DnaStatistX output into marginalized likelihoods per genotype per donor.
        """
        marg_likelihood_per_locus_genotype = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for line in process.stdout:
            if "DECONVOLUTION" in line and 'LocusName' not in line and 'H2' in line:
                marg_likelihood_per_locus_genotype = self.update_marg_lh_with_line(line,
                                                                                   marg_likelihood_per_locus_genotype)
        return marg_likelihood_per_locus_genotype

    def update_marg_lh_with_line(self, line: str, marg_lh: dict):
        genotypes, likelihood, locus = self.parse_line_to_values(line)
        for i_donor, locus_genotype in enumerate(genotypes):
            # precompute the marginal likelihoods for every donor D_i with genotype 'locus_genotype' g in
            # both mixtures: P(M|D_i=g)*P(D_i=g) per D_i (for all g)
            marg_lh[i_donor][locus][tuple(sorted(locus_genotype))] += likelihood
        return marg_lh

    @staticmethod
    def parse_line_to_values(line: str) -> tuple[list[tuple[str, str]], float, str]:
        line = line.replace('Ø', 'Q')
        line = line.split('\t')
        (_, _, locus), alleles, likelihood = line[:3], line[3:-1], line[-1]
        locus = locus.replace(' ', '')
        genotypes = list(zip(alleles[::2], alleles[1::2]))  # split flat list of alleles into tuples of two
        return genotypes, float(likelihood), locus

    def construct_cmd(self, trace_file: Path, n_contributors: int) -> list[str]:
        cmd = [r"C:\Program Files\Java\jdk-17\bin\java", "-jar", self.jar_file,
               "calculate",
               "--trace-profile", trace_file,
               "--reference-profile", self.reference_file,
               "--output", self.output_file,
               f"--{self.method}",
               f"--{self.model}",
               "-H1", "--contributors", str(n_contributors), "--cond-knowns", self._sin,
               "-H2", "--contributors", str(n_contributors),
               "--kit", self.kit,
               "-P", self.frequencies,
               "--threads", str(self.threads),
               "--coancestry 0.0",]
        # add all the thresholds specifically
        cmd += ["--threshold"]
        cmd.extend(self.thresholds)
        cmd.append(f"*={self.default_threshold}")
        if self.model == Models.EUROFORMIX.value:
            cmd.append("--degradation")
        return cmd

    def cleanup_output_files(self):
        for f in [f"{self.output_file}.json", f"{self.output_file}.txt"]:
            if os.path.exists(f):
                os.remove(f)

    @property
    def output_file(self) -> str:
        return self._output_file

    @output_file.setter
    def output_file(self, output_file: str):
        self._output_file = output_file

    @property
    def _sin(self):
        with open(self.reference_file, 'r') as f:
            f.readline()
            header = f.readline()
        sins = re.findall(SIN_PATTERN, header)
        return sins[0]

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['dnastatistx']

thresholds_config = Configuration(config['thresholds'])
dnax = DNAStatistX(
    jar_file=config['jar_file'],
    frequencies_file=config['frequencies_file'],
    model=config['model'],
    kit=config['kit'],
    threads=config['threads'],
    thresholds=thresholds_config,
    method=config['method']
)

def clean_dict_from_marg0(marg_lh: dict) -> dict:
    # Reformat a raw marginal likelihood dict into a clean per-locus dictionary
    cleaned_dict = {}
    for locus in marg_lh[0]:  # donor 0 is major donor 1 is minor in a 2p mixture
        cleaned_dict[locus] = {}
        for alleles, likelihood in marg_lh[0][locus].items():
            cleaned_dict[locus][alleles] = likelihood
    return cleaned_dict

def clean_dict_from_marg1(marg_lh: dict) -> dict:
    # Reformat a raw marginal likelihood dict into a clean per-locus dictionary
    cleaned_dict = {}
    for locus in marg_lh[1]:  # donor 0 is major donor 1 is minor in a 2p mixture
        cleaned_dict[locus] = {}
        for alleles, likelihood in marg_lh[1][locus].items():
            cleaned_dict[locus][alleles] = likelihood
    return cleaned_dict

def clean_dict_from_marg2(marg_lh: dict) -> dict:
    # Reformat a raw marginal likelihood dict into a clean per-locus dictionary
    cleaned_dict = {}
    for locus in marg_lh[2]:  # donor 0 is major donor 1 is minor in a 2p mixture
        cleaned_dict[locus] = {}
        for alleles, likelihood in marg_lh[2][locus].items():
            cleaned_dict[locus][alleles] = likelihood
    return cleaned_dict

#Which kit are we using ? Here are some options

five_loci = ['D3S1358', 'FGA', 'D18S51', 'vWA', 'TH01']
SGM = ['FGA', 'TH01', 'vWA', 'D2S1338', 'D3S1358', 'D8S1179', 'D16S539', 'D18S51', 'D19S433', 'D21S11']
NGM = ['D10S1248', 'vWA', 'D16S539', 'D2S1338', 'D8S1179', 'D21S11', 'D18S51', 'D22S1045', 'TH01', 'FGA', 'D2S441', 'D3S1358', 'D1S1656', 'D12S391', 'SE33']
all_loci = [
    'D1S1656', 'TPOX', 'D2S441', 'D2S1338', 'D3S1358', 'FGA', 'D5S818', 'CSF1PO',
    'SE33', 'D7S820', 'D8S1179', 'D10S1248', 'TH01', 'vWA', 'D12S391', 'D13S317',
    'PentaE', 'D16S539', 'D18S51', 'D19S433', 'PentaD', 'D21S11', 'D22S1045'
]

def make_allele_list(marg_lh: dict, locus: str) -> list:
    #Returns all alleles on a specific locus found in the mixture.
    allele_list = []
    for key in marg_lh[locus]:
        if (key[0] not in allele_list) and (key[0] != 'Q'):
            allele_list.append(key[0])
        elif (key[1] not in allele_list) and (key[1] != 'Q'):
            allele_list.append(key[1])
    return allele_list

def compute_freq(marg_lh: dict, locus: str, allele: tuple[str, str]) -> float:
    # This function determines the frequency of the two alleles on a locus. It uses the file NFI_frequencies.
    # It assumes independency of the alleles.
    freq = pd.read_csv('NFI_frequencies.csv', index_col=[0], encoding='ISO-8859-1')
    allele_list = make_allele_list(marg_lh, locus)
    p_q = 1
    for allelel in allele_list:
        p_q -= freq[locus][float(allelel)]
    if (allele[0] in allele_list) and (allele[1] in allele_list):
        presenceprob = freq[locus][float(allele[0])]*freq[locus][float(allele[1])]
    elif (allele[0] not in allele_list) and (allele[1] in allele_list):
        presenceprob = p_q*freq[locus][float(allele[1])]
    elif (allele[0] in allele_list) and (allele[1] not in allele_list):
        presenceprob = freq[locus][float(allele[0])]*p_q
    elif (allele[0] not in allele_list) and (allele[1] not in allele_list):
        presenceprob = p_q*p_q
    return presenceprob

def dict_with_freq_alleles(marg_lh: dict, locus: str, allele_list: list) -> dict:
    #This function returns a dictionary with all frequencies for possible allele combinations.
    freq = pd.read_csv('NFI_frequencies.csv', index_col=[0], encoding='ISO-8859-1')
    freq_dict = {}
    allele_list_with_Q = allele_list + ['Q']
    for i in allele_list_with_Q:
        for j in allele_list_with_Q:
            string = i + '+' + j
            freq_dict[string] = float(compute_freq(marg_lh, locus, (i,j)))
    return freq_dict

def prob_E_known_g_u(marg_lh: dict, locus: str) -> float:
    #= \sum P(g_i)P(E|D_1=g+i)
    allele_list = make_allele_list(marg_lh, locus)
    allele_list_with_Q = allele_list + ['Q']
    P_E_known_gu = 0
    freq_dict = dict_with_freq_alleles(marg_lh, locus, allele_list)
    for i in allele_list_with_Q:
        for j in allele_list_with_Q:
            string = i + '+' + j
            if (i,j) in marg_lh[locus]:
                P_E_known_gu += freq_dict[string] * marg_lh[locus][(i,j)]
    return P_E_known_gu

def LR_g1_gu(marg_lh: dict, locus: str) -> dict:
    # Calculates LRs for all allele combinations possible on a locus
    allele_list = make_allele_list(marg_lh, locus)
    allele_list_with_Q = allele_list + ['Q']
    P_E_known_gu = prob_E_known_g_u(marg_lh, locus)
    dict3 = {}
    for i in allele_list_with_Q:
        for j in allele_list_with_Q:
            if (i,j) in marg_lh[locus]:
                dict3[(i,j)] = marg_lh[locus][(i,j)]/P_E_known_gu
    return dict3


def calculate_all_99_posterior(marg_lh: dict, loci_list: list, threshold: float) -> dict:
    posterior_99_dict = {}
    cum_sum_list = []
    for locus in loci_list:
        locus_data = marg_lh[locus]
        total_posterior = sum(locus_data.values())
        posterior_probs = {}
        for genotype, value in locus_data.items():
            posterior_probs[genotype] = value / total_posterior

        sorted_genotypes = sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)

        cum_sum = 0
        top_genotypes = {}
        for genotype, prob in sorted_genotypes:
            cum_sum += prob
            top_genotypes[genotype] = prob
            if cum_sum >= threshold:
                cum_sum_list.append(cum_sum)
                break
        posterior_99_dict[locus] = top_genotypes
    prob_in_histogram = math.prod(cum_sum_list)
    return posterior_99_dict, prob_in_histogram


def compute_true_LRs(marg_lh: dict, locus: str) -> dict:
    allele_list = make_allele_list(marg_lh, locus)
    freq_dict = dict_with_freq_alleles(marg_lh, locus, allele_list)

    numerators = {}
    for g in marg_lh[locus]:
        p_g = freq_dict[g[0] + '+' + g[1]]
        if p_g > 0:
            numerators[g] = marg_lh[locus][g] / p_g

    denominator = sum(marg_lh[locus].values())
    return {g: numerators[g] / denominator for g in numerators}

from itertools import product

def compute_full_LRs_from_top_posteriors(marg_lh: dict, posterior_99: dict) -> dict:

    per_locus_LRs = {}
    loci = list(posterior_99.keys())

    for locus in loci:
        true_lrs = compute_true_LRs(marg_lh, locus)
        per_locus_LRs[locus] = {
            g: true_lrs[g] for g in posterior_99[locus] if g in true_lrs
        }
    genotype_options = [list(per_locus_LRs[locus].keys()) for locus in loci]
    full_LRs = {}
    for full_genotype in product(*genotype_options):
        product_lr = 1
        for locus, g in zip(loci, full_genotype):
            product_lr *= per_locus_LRs[locus][g]
        full_LRs[full_genotype] = product_lr

    return full_LRs

def compute_full_LRs_from_top_posteriors2(marg_lh: dict, loci_list: list, sampled_genotypes: list) -> list:
    LR_per_locus = {}
    for locus in loci_list:
        LR_per_locus[locus] = compute_true_LRs(marg_lh, locus)

    full_LRs = []
    for genotype in sampled_genotypes:
        LR = 1
        for locus, alleles in zip(loci_list, genotype):
            if alleles in LR_per_locus[locus]:
                LR *= LR_per_locus[locus][alleles]
            elif tuple(reversed(alleles)) in LR_per_locus[locus]:
                # Fallback: try reversed allele order
                LR *= LR_per_locus[locus][tuple(reversed(alleles))]
            else:
                # If neither order is present, set LR to 0
                LR *= 0
        full_LRs.append(LR)
    return full_LRs
# def compute_full_genotype_posteriors(posterior_99: dict) -> dict:
#     loci = list(posterior_99.keys())
#     genotype_combinations = product(*(posterior_99[locus].items() for locus in loci))
#     full_posteriors = {}
#     for genotype_tuple in genotype_combinations:
#         full_genotype = tuple(g for g, _ in genotype_tuple)
#         prob = math.prod(prob for _, prob in genotype_tuple)
#         full_posteriors[full_genotype] = prob
#     return full_posteriors

import numpy as np
from itertools import product


def compute_full_genotype_posteriors_fast(posterior_99: dict) -> dict:

    loci = list(posterior_99.keys())
    genotype_lists = [list(posterior_99[locus].items()) for locus in loci]

    keys_per_locus = [np.array([g for g, _ in lst], dtype=object) for lst in genotype_lists]
    probs_per_locus = [np.array([p for _, p in lst]) for lst in genotype_lists]

    mesh = np.meshgrid(*probs_per_locus, indexing='ij')
    prob_matrix = np.stack(mesh, axis=-1)
    joint_probs = np.prod(prob_matrix, axis=-1).flatten()

    key_product = list(product(*keys_per_locus))

    full_posteriors = {
        tuple(tuple(g) for g in genotype_combo): prob
        for genotype_combo, prob in zip(key_product, joint_probs)
    }

    return full_posteriors



def make_histogram(log10_lrs, name_plot, true_LR):
    plt.figure(figsize=(10, 6))
    plt.hist(log10_lrs, bins=100, alpha = 0.5, edgecolor='black', label = 'generated donors')
    plt.title("Histogram of log₁₀(LR) with the threshold method", fontsize = 18)
    plt.xlabel("log₁₀(LR)", fontsize = 18)
    plt.ylabel("Frequency", fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.grid(True)
    plt.axvline(x=true_LR, color='red', linestyle='--', linewidth=2,
                label=f'true-donor LR = {true_LR}')
    plt.legend(loc = 'upper left',fontsize = 18)
    plt.tight_layout()
    plt.savefig(f'plots/{name_plot}_histogram.png')
    plt.show()

def make_posterior_histogram(log10_lrs, posterior_probs, name_plot):
    plt.figure(figsize=(10, 6))
    plt.scatter(log10_lrs, posterior_probs, alpha=0.5, edgecolor='black', label = 'generated donors')
    plt.title("log₁₀(LR) vs Posterior Probability with the threshold method", fontsize = 18)
    plt.xlabel("log₁₀(LR)", fontsize = 18)
    plt.ylabel("Posterior Probability", fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.grid(True)
    plt.legend(loc = 'upper left',fontsize = 18)
    plt.tight_layout()
    plt.savefig(f'plots/{name_plot}_posterior.png')
    plt.show()


def plot_posterior_with_cutoff(log10_lrs, posterior_probs, name_plot, threshold=0.5):
    data = sorted(zip(log10_lrs, posterior_probs), key=lambda x: x[0], reverse=True)
    cum_sum = 0
    log_cutoff = None
    for log_lr, post in data:
        cum_sum += post
        if cum_sum >= threshold:
            log_cutoff = log_lr
            break
    print(log_lr)
    plt.figure(figsize=(10, 6))
    plt.scatter(log10_lrs, posterior_probs, alpha=0.5, edgecolor='black', label = 'generated donors')
    plt.title("log₁₀(LR) vs Posterior Probability with the threshold method", fontsize = 18)
    plt.xlabel("log₁₀(LR)", fontsize = 18)
    plt.ylabel("Posterior Probability", fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{name_plot}_posteriorscatter.png')
    # Add cutoff line
    if log_cutoff is not None:
        plt.axvline(x=log_cutoff, color='red', linestyle='--', linewidth=2,
                    label=f'cutoff ≈ {log_cutoff:.4f}, prob = {threshold}')
        plt.legend(fontsize=18)

    plt.tight_layout()
    plt.savefig(f'plots/{name_plot}_posteriorscatter_cutoff.png', dpi=300)
    plt.show()

    lr_cutoff = 10 ** log_cutoff if log_cutoff is not None else None
    return log_cutoff, lr_cutoff

def format_allele(value):
    # Convert float to int if it's a whole number
    if float(value).is_integer():
        return str(int(value))
    return str(value)


llh_estimate, marg_likelihood_per_locus_genotype = dnax.run(Path("Mixtures/1_1C2.txt"), 2) ### Select the mixture and the number of contributors
cleanmarg = clean_dict_from_marg0(marg_likelihood_per_locus_genotype)      ### Change here the number to the correct donor
posterior_99, prob_hist = calculate_all_99_posterior(cleanmarg, all_loci, threshold=0.75)  ## Change the threshold
print(posterior_99)
full_LRs = compute_full_LRs_from_top_posteriors(cleanmarg, posterior_99)
lr_values = list(full_LRs.values())
log10_lrs = np.log10(lr_values)
df = pd.read_csv(r'U:\RIHAL-admin\PycharmProjects\PythonProject\.venv\1A.csv', delimiter=';')  ## Select the true donor
true_genotype = tuple((format_allele(a1), format_allele(a2)) for a1, a2 in zip(df['Allele1'], df['Allele2']))  # Not with Q yet
true_LR = compute_full_LRs_from_top_posteriors2(cleanmarg, all_loci, [true_genotype])
log_true_LR = math.log10(true_LR[0])
make_histogram(log10_lrs, 'C2_major_0.75', log_true_LR)
print(f'The probability of the true genotype being in the histogram is: {prob_hist}', flush = True)
import time; time.sleep(5)
posteriors = compute_full_genotype_posteriors_fast(posterior_99).values()
make_posterior_histogram(log10_lrs, posteriors, 'C2_major_0.75')
#log_cutoff, lr_cutoff = plot_posterior_with_cutoff(log10_lrs, posteriors, name_plot='A2_minor_0.95')
#print(lr_cutoff)
# Print top LR combinations
#top = sorted(full_LRs.items(), key=lambda x: x[1], reverse=True)
#for combo, lr in top:
#    print(f"{combo} => Full LR: {lr:.4e}")
#print(len(top))

# Extract LR values and compute log10(LR)

# your code
print(f"Time taken: {time.time() - start:.2f}s")
