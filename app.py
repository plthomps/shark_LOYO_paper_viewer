from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import fnmatch
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT=Path('results')
APP_DIR=Path(__file__).resolve().parent
ARCHIVES=APP_DIR/'shark_charts'

# Notebook ZIP members start with analysis_2022/. Support repositories
# that contain a nested shark_charts folder and archives with an outer folder.
def available_archives():
    candidates=list(ARCHIVES.glob('*.zip')) if ARCHIVES.is_dir() else []
    if not candidates:
        candidates=list(APP_DIR.rglob('*.zip'))
    return sorted(set(p.resolve() for p in candidates))

def canonical_member(name):
    name=name.replace('\\','/').lstrip('/')
    for root in ('analysis_2022/','results/'):
        if name.startswith(root):
            return name
        marker='/'+root
        if marker in name:
            return root+name.split(marker,1)[1]
    return name

@st.cache_resource(show_spinner='Indexing figure archives…')
def archive_index(signature):
    index={}
    for archive_name,_,_ in signature:
        archive=Path(archive_name)
        with ZipFile(archive) as z:
            for entry in z.namelist():
                if entry.endswith('/'):
                    continue
                key=canonical_member(entry)
                if key in index:
                    raise ValueError(f'Duplicate archive member: {key}')
                index[key]=(archive,entry)
    return index

def archive_signature():
    return tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in available_archives())

def current_index():
    return archive_index(archive_signature())

def member_path(path):
    return path.as_posix()

def exists(path):
    return member_path(path) in current_index()

def read_bytes(path):
    name=member_path(path)
    found=current_index().get(name)
    if found is None:
        raise FileNotFoundError(name)
    archive,actual_name=found
    with ZipFile(archive) as z:
        return z.read(actual_name)

def glob_members(folder,pattern):
    prefix=member_path(folder).rstrip('/')+'/'
    return sorted(Path(k) for k in current_index() if k.startswith(prefix)
                  and '/' not in k[len(prefix):] and fnmatch.fnmatch(k[len(prefix):],pattern))
PERIODS=['ALL','DAWN','DAY','DUSK','NIGHT']
st.set_page_config(page_title='Cape Cod Shark | LOYO Paper Figures',layout='wide')
st.title('Cape Cod white shark · LOYO paper figures')
st.caption('LOYO validation 2017–2024; full-data held-out 2022 partial dependence, nonadditive interactions and Friedman H. Figures are read directly from ZIP archives; no retraining.')
if not current_index():
    st.error('No ZIP archives found in shark_charts/. Upload all seven github_upload_batch_XX.zip files to that folder on GitHub.')
    st.stop()
st.sidebar.caption(f'{len(current_index()):,} archived files · {len(available_archives())} ZIP archives')

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

section=st.sidebar.radio('Section',['Overview & performance','Temporal heatmaps','Feature importance','Partial dependence','SHAP','Two-way PDPs','Nonadditive interactions & H','Explanations','Paper figure gallery'])
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
elif section=='Explanations':
    st.subheader('Understanding partial dependence and interactions')
    st.markdown("""
    ### Imagine warm water and calm winds
    We want to understand what the Random Forest predicts about shark presence.

    **One-way PDP:** How does predicted shark presence change with water temperature, averaging over other environmental conditions?

    **Two-way PDP:** What does the model predict for **each combination** of water temperature and wind stress? This is the **joint predicted response**. Two strong separate effects can create a striking contour plot even when there is **no interaction**.

    **Two-way interaction surface:** Remove the individual temperature effect and individual wind effect from the joint PDP. **What remains** is the nonadditive response: does wind behave differently at different temperatures?

    **Friedman's H:** A number describing how large that leftover is relative to the overall variation in the two-way PDP. Values near zero suggest little model interaction. Higher values indicate stronger nonadditivity on the evaluated grid. **H is not a p-value**, and there are no universal significance cutoffs.

    #### What does *nonadditive* mean?
    **Additive:** Imagine a baseline predicted shark presence of **20%**. Warm water adds **20 percentage points**, and calm winds add **10 percentage points**. Together, we would expect **20% + 20% + 10% = 50%**.

    **Nonadditive:** But what if the model predicts **75%** when water is warm **and** winds are calm? That is **25 percentage points above** the additive expectation. The combination behaves differently from simply adding the separate effects: the effect of one condition depends on the other.

    | Illustrative model scenario | Predicted shark presence |
    |---|---:|
    | Baseline | 20% |
    | Warm water alone | 40% |
    | Calm winds alone | 30% |
    | Warm water + calm winds, if additive | **50%** |
    | Warm water + calm winds, actual model prediction | **75%** |
    | **Nonadditive departure** | **+25 percentage points** |

    *These numbers are examples, not findings from our shark models.* A nonadditive departure can also be negative. The actual interaction surfaces calculate the additive expectation using the full PDP grid, not just a single baseline. They describe **model behavior**, not biological causation.

    #### A simple example
    If calm winds increase predicted shark presence by **10 percentage points** in both cold and warm water, there is **no interaction**: the wind effect is unchanged.

    If calm winds increase predicted presence by **5 points in cold water** but **35 points in warm water**, there **is an interaction**: wind's effect depends on temperature.

    #### How to read the figures
    - **PDP contour:** The colors represent average predicted shark-presence probability.
    - **Interaction contour:** Red and blue show *positive and negative departures from an additive prediction*. These are probability **differences**, not probabilities.
    - **H ranking:** Helps prioritize which predictor combinations to investigate. Inspect the **absolute interaction magnitude** too: a large relative H can occur even when the model's joint response barely changes.

    #### Important scientific limitations
    These figures describe the fitted Random Forest, not causality. Correlated predictors can produce unrealistic combinations on PDP grids. Contour lines interpolate between calculated grid points, and the H values here use a **uniform-grid approximation** on the seven-point-per-axis PDP grid. H does not establish statistical significance. A plot labeled 'two-way interaction' in older work may actually be a two-way PDP; the separate nonadditive surface makes the distinction explicit.
    """)
elif section=='Nonadditive interactions & H':
    st.subheader('LOYO 2022: Friedman H and nonadditive interaction plots')
    root=Path('analysis_2022')
    summary_path=root/'friedman_h_all_2022.csv'
    if not exists(summary_path):
        st.error('The Friedman H summary is not visible in the loaded ZIP archives. The analysis does not need to be rerun.')
        interaction_zips=[p for p in available_archives() if p.name.startswith('interaction_analysis_batch_')]
        st.write(f'Interaction ZIP files detected: **{len(interaction_zips)} of 8 expected**')
        st.write('Detected ZIP filenames:', [p.name for p in interaction_zips])
        matches=[k for k in current_index() if 'friedman_h_all_2022.csv' in k]
        st.write('Matching summary paths found inside ZIPs:', matches if matches else 'None')
        st.write('Expected archive member:', str(summary_path))
        st.write('Archive search directory:', str(ARCHIVES))
        st.write('All ZIPs found:', [str(p.relative_to(APP_DIR)) if p.is_relative_to(APP_DIR) else str(p) for p in available_archives()])
        st.write('Sample indexed interaction paths:', [k for k in current_index() if 'interaction_2way/' in k][:5])
        st.caption('Confirm all eight ZIPs are committed to shark_charts/ in the deployed GitHub branch. If they are, reboot the Streamlit app.')
    else:
        summary=load_csv(summary_path)
        period_table=summary.loc[summary.period==period].sort_values('friedman_h',ascending=False,na_position='last')
        st.caption('Uniform-grid, descriptive Friedman H based on full held-out 2022 observations (no row sampling). H is not a p-value; missing H indicates an undefined value for a nearly constant joint surface.')
        st.dataframe(period_table[['feature_x','feature_y','friedman_h','n_reference_rows','grid_x','grid_y']],use_container_width=True,hide_index=True)
        st.download_button('Download all 375 H values',read_bytes(summary_path),file_name='friedman_h_all_2022.csv')
        if len(period_table):
            choices=[f'{r.feature_x} × {r.feature_y}' for r in period_table.itertuples(index=False)]
            selected=st.selectbox('Predictor pair (ranked by H)',choices)
            a,b=selected.split(' × ',1)
            chosen=period_table.loc[(period_table['feature_x'].astype(str)==a) & (period_table['feature_y'].astype(str)==b)].iloc[0]
            h_value=pd.to_numeric(chosen['friedman_h'],errors='coerce')
            rank_position=choices.index(selected)+1
            interaction_csv=Path('analysis_2022')/period/'interaction_2way'/f'{a}__{b}.csv'
            magnitude_pp=None
            if exists(interaction_csv):
                interaction_values=load_csv(interaction_csv)
                if 'nonadditive_probability' in interaction_values.columns:
                    values=pd.to_numeric(interaction_values['nonadditive_probability'],errors='coerce').dropna()
                    if not values.empty:
                        magnitude_pp=float(values.abs().max()*100)
            st.markdown(f'#### {period}: {a} × {b}')
            m1,m2,m3=st.columns(3)
            m1.metric('Friedman H',f'{h_value:.4f}' if pd.notna(h_value) else 'Undefined')
            m2.metric('H rank within '+period,f'{rank_position} / {len(period_table)}')
            m3.metric('Maximum |nonadditive effect|',f'{magnitude_pp:.3f} pp' if magnitude_pp is not None else 'Unavailable')
            st.caption('H is a relative, descriptive measure of nonadditivity (not a p-value). The maximum absolute effect is the largest departure from additivity on the evaluated grid, expressed in percentage points (pp). These measures do not establish causation.')
            first=Path('analysis_2022')/period/'pdp_2way'/f'{a}__{b}.png'
            second=Path('analysis_2022')/period/'interaction_2way'/f'{a}__{b}.png'
            c1,c2=st.columns(2)
            with c1:
                st.markdown('**Two-way PDP: joint predicted probability**')
                figure(first)
            with c2:
                st.markdown('**Interaction: departure from additivity**')
                figure(second)
            csv_view(second.with_suffix('.csv'))
        heatmap=root/'friedman_h_heatmap.png'
        if exists(heatmap):
            with st.expander('Cross-diel Friedman H heatmap'):
                figure(heatmap)
                csv_view(root/'friedman_h_heatmap_values.csv')
elif section=='Two-way PDPs':
    st.subheader(f'{period}: two-way partial dependence')
    st.caption('These original figures show joint responses. For nonadditive interaction surfaces and Friedman H, choose Nonadditive interactions & H from the sidebar.')
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
    folders={'Cross-diel comparisons':ROOT,'2022 performance':base/'paper_figures','2022 PDP':base/'pdp_1way','2022 SHAP':base/'shap_dependence_18','2022 original two-way PDPs':base/'pdp_2way','2022 nonadditive interactions':Path('analysis_2022')/period/'interaction_2way','2022 new PDP contours':Path('analysis_2022')/period/'pdp_2way'}
    folder=st.selectbox('Figure group',list(folders));files=sorted(glob_members(folders[folder],'*.png'))
    if files:
        search=st.text_input('Filter figure names');files=[p for p in files if search.lower() in p.stem.lower()]
        for p in files[:30]:figure(p)
        if len(files)>30:st.info(f'Showing first 30 of {len(files)}; filter to narrow.')
    else:st.info('No figures in this group.')
