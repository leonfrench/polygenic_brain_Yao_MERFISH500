"""HomoloGene conversion to mouse symbols using the SEA-AD app's supplied mapping snapshot."""
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd

MAPPING_PATH = Path(__file__).resolve().parents[1]/'data/homologene_to_human.csv.gz'
SPECIES = ('Mouse', 'Human', 'Rhesus Macaque')


@lru_cache(maxsize=1)
def load_ortholog_map():
    """Return {species: {source symbol: {human symbols}}} and {HUMAN: {mouse symbols}}."""
    frame = pd.read_csv(MAPPING_PATH, dtype=str, keep_default_na=False)
    if not {'species', 'source_symbol', 'human_symbol'}.issubset(frame.columns):
        raise ValueError('Ortholog mapping is missing required columns.')
    to_human, human_to_mouse = {}, {}
    for species, source, human in frame[['species', 'source_symbol', 'human_symbol']].itertuples(index=False, name=None):
        to_human.setdefault(species, {}).setdefault(source, set()).add(human)
        if species == 'Mouse':
            human_to_mouse.setdefault(human.upper(), set()).add(source)
    return to_human, human_to_mouse


def map_input(text, species):
    """Return deduplicated mouse symbols and source symbols without a mouse ortholog.

    Human symbols match case-insensitively (HGNC symbols are upper case). Rhesus
    symbols are case-sensitive, as in the reference app, and pass through their
    human orthologs. One-to-many orthologs are all retained before deduplication.
    """
    if species not in SPECIES:
        raise ValueError(f'Unsupported species: {species}')
    requested = sorted(set(filter(None, re.split(r'[\s,;]+', text.strip()))))
    if species == 'Mouse':
        return requested, []
    to_human, human_to_mouse = load_ortholog_map()
    if species == 'Human':
        humans = {gene: {gene.upper()} for gene in requested}
    else:
        humans = {gene: to_human[species].get(gene, set()) for gene in requested}
    mouse = {gene: {m for h in humans[gene] for m in human_to_mouse.get(h.upper(), ())} for gene in requested}
    return sorted(set().union(*mouse.values())), [gene for gene in requested if not mouse[gene]]
