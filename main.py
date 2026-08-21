import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



df_fake=pd.read_csv("/home/gabriel/Obsidian Vault/​🗂️​ FACULDADE/🗂️ 4° Semestre/Projeto Integrador/Codigo/DF/Fake.csv")
df_true=pd.read_csv("/home/gabriel/Obsidian Vault/​🗂️​ FACULDADE/🗂️ 4° Semestre/Projeto Integrador/Codigo/DF/True.csv")