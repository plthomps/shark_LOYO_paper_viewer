from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import fnmatch
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT=Path('results')
ARCHIVES=Path(__file__).parent/'shark_charts'

@st.cache_resource(show_spinner='Indexing figure archives…')
def archive_index():
    archives=sorted(ARCHIVES.glob('*.zip'))
    index={}
    for archive in archives:
        with ZipFile(archive) as z:
            for entry in z.namelist():
                if not entry.endswith('/'):
                    if entry in index:
                        raise ValueError(f'Duplicate archive member: {entry}')
                    index[entry]=archive
    return index

def member_path(path):
    return path.as_posix()

def exists(path):
    return member_path(path) in archive_index()

def read_bytes(path):
    name=member_path(path)
    archive=archive_index().get(name)
    if archive is None:
        raise FileNotFoundError(name)
    with ZipFile(archive) as z:
        return z.read(name)

def glob_members(folder,pattern):
    prefix=member_path(folder).rstrip('/')+'/'
    return sorted(Path(k) for k in archive_index() if k.startswith(prefix)
                  and '/' not in k[len(prefix):] and fnmatch.fnmatch(k[len(prefix):],pattern))
PERIODS=['ALL','DAWN','DAY','DUSK','NIGHT']
st.set_page_config(page_title='Cape Cod Shark | LOYO Paper Figures',layout='wide')
st.title('Cape Cod white shark · LOYO paper figures')
st.caption('LOYO validation 2017–2024; detailed held-out 2022 interpretability. Figures are read directly from ZIP archives, not live retraining.')
if not archive_index():
    st.error('No ZIP archives found in shark_charts/. Upload all seven github_upload_batch_XX.zip files to that folder on GitHub.')
    st.stop()
st.sidebar.caption(f'{len(archive_index()):,} archived files · {len(list(ARCHIVES.glob("*.zip")))} ZIP archives')

@st.cache_data
def load_csv(path):
    return pd.read_csv(BytesIO(read_bytes(path)))

def figure(path,caption=None):
    if not exists(path):
        st.info('Figure unavailable: '+path.name);return
    st.image(read_bytes(path),caption=caption or path.stem.replace('_',' '),use_container_width=True)
    a,b=st.columns(2)
    with a:
        st.download_button('Download PNG',read_bytes(path),file_name=path.name,key=str(path))
    pdf=path.with_suffix('.pdf')
    if exists(pdf):
        with b:
            st.download_button('Download PDF',read_bytes(pdf),file_name=pdf.name,key=str(pdf))

def csv_view(path):
    if exists(path):
        df=load_csv(path);st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button('Download CSV',df.to_csv(index=False).encode(),file_name=path.name,key='csv'+str(path))

section=st.sidebar.radio('Section',['Overview & performance','Temporal heatmaps','Feature importance','Partial dependence','SHAP','Two-way interactions','Paper figure gallery'])
period=st.sidebar.selectbox('2022 diel model',PERIODS)
base=ROOT/'LOYO'/f'2022_{period}'

if section=='Overview & performance':
    st.subheader('2022 held-out performance')
    csv_view(ROOT/'table_2022_all_diel_metrics.csv')
    st.subheader(f'{period}: performance panel')
    figure(base/'paper_figures'/'performance_panel.png')
    st.subheader('All LOYO years')
    csv_view(ROOT/'loyo_metrics_all_years.csv')
elif section=='Temporal heatmaps':
    st.subheader('LOYO AUC by year and diel period')
    figure(ROOT/'heatmap_auc_year_diel.png')
    csv_view(ROOT/'heatmap_auc_values.csv')
    st.subheader('Average precision by year and diel period')
    figure(ROOT/'heatmap_ap_year_diel.png')
    csv_view(ROOT/'heatmap_ap_values.csv')
elif section=='Feature importance':
    st.subheader(f'{period}: 18-feature importance')
    for name in ['gini_importance','permutation_importance','shap_importance_18']:
        st.markdown('#### '+name.replace('_',' ').title())
        figure(base/'paper_figures'/f'{name}.png')
        csv_view(base/f'{name}.csv')
    st.subheader('Across-diel SHAP importance')
    figure(ROOT/'shap_importance_heatmap_2022.png')
    csv_view(ROOT/'shap_importance_heatmap_values.csv')
elif section=='Partial dependence':
    st.subheader(f'{period}: one-way PDP')
    figs=sorted(glob_members(base/'pdp_1way','*.png'))
    if figs:
        feature=st.selectbox('Predictor',[f.stem for f in figs]);p=base/'pdp_1way'/f'{feature}.png'
        figure(p);csv_view(p.with_suffix('.csv'))
    else:st.warning('No PDP files found.')
elif section=='SHAP':
    st.subheader(f'{period}: SHAP')
    for name in ['shap_importance_18','shap_beeswarm_encoded']:
        figure(base/'paper_figures'/f'{name}.png')
    figs=sorted(glob_members(base/'shap_dependence_18','*.png'))
    if figs:
        feature=st.selectbox('SHAP dependence predictor',[f.stem for f in figs]);figure(base/'shap_dependence_18'/f'{feature}.png')
    csv_view(base/'shap_importance_18.csv')
elif section=='Two-way interactions':
    st.subheader(f'{period}: two-way partial dependence')
    figs=sorted(glob_members(base/'pdp_2way','*.png'))
    if figs:
        first=sorted(set(f.stem.split('__')[0] for f in figs if '__' in f.stem))
        selected=st.selectbox('First predictor',first)
        candidates=[f for f in figs if f.stem.startswith(selected+'__')]
        second=st.selectbox('Second predictor',[f.stem.split('__',1)[1] for f in candidates])
        p=base/'pdp_2way'/f'{selected}__{second}.png';figure(p);csv_view(p.with_suffix('.csv'))
        st.caption('Two-way PDPs show joint model responses; they do not by themselves establish a non-additive interaction or causation.')
    else:st.warning('No interaction plots found.')
else:
    st.subheader('Paper figure gallery')
    folders={'Cross-diel comparisons':ROOT,'2022 performance':base/'paper_figures','2022 PDP':base/'pdp_1way','2022 SHAP':base/'shap_dependence_18','2022 interactions':base/'pdp_2way'}
    folder=st.selectbox('Figure group',list(folders));files=sorted(glob_members(folders[folder],'*.png'))
    if files:
        search=st.text_input('Filter figure names');files=[p for p in files if search.lower() in p.stem.lower()]
        for p in files[:30]:figure(p)
        if len(files)>30:st.info(f'Showing first 30 of {len(files)}; filter to narrow.')
    else:st.info('No figures in this group.')
