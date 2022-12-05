import os
import random
import numpy as np
import pandas as pd

from CGRtools import SDFRead, SDFWrite
from CGRtools import RDFRead, RDFWrite
from CIMtools.preprocessing.conditions_container import DictToConditions, ConditionsToDataFrame
from CIMtools.preprocessing import Fragmentor, CGR, EquationTransformer, SolventVectorizer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from os import environ
fragmentor_path='./colearn'
environ['PATH'] += ':{}'.format(fragmentor_path)


import os
import random
import shutil
import numpy as np
import pandas as pd

from CGRtools import SDFRead, SDFWrite
from CGRtools import RDFRead, RDFWrite
from CIMtools.preprocessing.conditions_container import DictToConditions, ConditionsToDataFrame
from CIMtools.preprocessing import Fragmentor, CGR, EquationTransformer, SolventVectorizer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from os import environ
fragmentor_path = './colearn'
environ['PATH'] += ':{}'.format(fragmentor_path)


def extract_meta(x):
    return [i[0].meta for i in x]

    
def mark_molecules(mols):
    mols_marked = []
    for m in mols:
        m_copy = m.copy()
        m_copy.atom(1).charge += 1
        cgr = m.compose(m_copy)
        
        cgr.meta.update(m.meta)
        mols_marked.append(cgr)
    return mols_marked


def mark_reactions(reacts):

    reagents_marked, products_marked = [], []
    for reaction in reacts:

        reagent, product = reaction.molecules()
        r_is_marked, p_is_marked = False, False
        for n, a in reagent.atoms():
            
            if a.implicit_hydrogens > product.atom(n).implicit_hydrogens:
                reagent_copy = reagent.copy()
                reagent_copy.atom(n).charge += 1
                cgr = reagent.compose(reagent_copy)
                cgr.meta.update(reaction.meta)
                reagents_marked.append(cgr)
                
                r_is_marked = True
                
            elif a.implicit_hydrogens < product.atom(n).implicit_hydrogens:
                product_copy = product.copy()
                product_copy.atom(n).charge += 1
                cgr = product.compose(product_copy)
                cgr.meta.update(reaction.meta)
                products_marked.append(cgr)
                
                p_is_marked = True
                
                #
        if (r_is_marked is False) and (p_is_marked is False):
            reagents_marked.append(reagent)
            products_marked.append(product)
                
    return reagents_marked, products_marked


class ISIDAFragmentor:
    
    def __init__(self, cgr=False, conditions=False):
        self.cgr = cgr
        self.conditions = conditions
        pass

   
    def fit(self, mols):
        #
        if self.cgr:
            graph = Pipeline([('CGR', CGR()), ('frg', Fragmentor(fragment_type=9, max_length=3, useformalcharge=True, version='2017.x'))])
        else:
            graph = Pipeline([('frg', Fragmentor(fragment_type=9, max_length=3, useformalcharge=True, version='2017.x'))])
        #
        if self.conditions:
     
            features = ColumnTransformer([('temp', EquationTransformer('1/x'), ['temperature']),
                                          ('solv', SolventVectorizer(), ['solvent.1']),
                                          ('amount', 'passthrough', ['solvent_amount.1']),])

            conditions = Pipeline([('meta', FunctionTransformer(extract_meta)),
                                   ('cond', DictToConditions(solvents=('additive.1',), temperature='temperature', amounts=('amount.1',))),
                                   ('desc', ConditionsToDataFrame()),
                                   ('final', features)])
            #
            self.frag = ColumnTransformer([('graph', graph, [0]), ('cond', conditions, [0])])
        else:
            self.frag = ColumnTransformer([('graph', graph, [0])])
        #
        self.frag.fit([[i] for i in mols])
        #
        del frag
        frg_files = [i for i in os.listdir() if i.startswith('frg')]
        for file in frg_files:
            if os.path.isfile(file):
                os.remove(file)
            else:
                shutil.rmtree(file)

        return 
    
    def transform(self, mols):
        x = self.frag.transform([[i] for i in mols])
        return x