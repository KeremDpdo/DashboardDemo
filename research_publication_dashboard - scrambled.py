import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import uuid

# Page configuration
st.set_page_config(page_title="Research Publications Dashboard (Demo)", layout="centered")

# Demo Disclaimer
st.markdown("""
    <div style='background-color: #FFF3CD; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <strong>Demo Notice:</strong> This is a demonstration of a research publications dashboard designed to process uploaded files from METU. For this demo, the data is randomly generated and does not represent real information. This dashboard was used internally to identify and visualize relevant trends.
    </div>
""", unsafe_allow_html=True)

# CSS for styling and full-screen fix
st.markdown("""
    <style>
        .main > div {
            max-width: 900px;
            margin: 0 auto;
            padding: 0 20px;
        }
        .stPlotlyChart, .js-plotly-plot, .plotly {
            width: 100% !important;
            max-width: 900px !important;
            margin: 0 auto;
        }
        /* Hide modebar in full-screen */
        .modebar {
            display: block !important;
        }
        .modebar-group .modebar-btn[data-title="Toggle fullscreen"] {
            display: none !important;
        }
        /* Full-screen mode styling */
        .js-plotly-plot:-webkit-full-screen, .js-plotly-plot:-moz-full-screen, .js-plotly-plot:fullscreen {
            width: 100vw !important;
            height: 100vh !important;
            max-width: 100% !important;
            max-height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            background-color: #fff !important;
        }
        /* Ensure parent container stretches */
        .stPlotlyChart:-webkit-full-screen, .stPlotlyChart:-moz-full-screen, .stPlotlyChart:fullscreen {
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* Reset Plotly's internal margins */
        .js-plotly-plot:-webkit-full-screen .plotly, .js-plotly-plot:-moz-full-screen .plotly, .js-plotly-plot:fullscreen .plotly {
            margin: 0 !important;
            padding: 0 !important;
        }
        /* Consistent font styling */
        body, h1, h2, h3, p, div, span {
            font-family: 'Roboto', sans-serif !important;
        }
        h1, h2, h3 {
            color: #1F77B4;
        }
    </style>
""", unsafe_allow_html=True)

# JavaScript to enhance full-screen behavior
st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const plots = document.querySelectorAll('.js-plotly-plot');
            plots.forEach(plot => {
                plot.on('plotly_fullscreen', function() {
                    plot.style.width = '100vw';
                    plot.style.height = '100vh';
                    plot.style.maxWidth = '100%';
                    plot.style.maxHeight = '100%';
                    plot.style.margin = '0';
                    plot.style.padding = '0';
                    plot.style.position = 'fixed';
                    plot.style.top = '0';
                    plot.style.left = '0';
                    plot.style.display = 'flex';
                    plot.style.justifyContent = 'center';
                    plot.style.alignItems = 'center';
                    // Hide modebar in full-screen
                    const modebar = plot.querySelector('.modebar');
                    if (modebar) modebar.style.display = 'none';
                });
                plot.on('plotly_fullscreen_exit', function() {
                    plot.style.width = '';
                    plot.style.height = '';
                    plot.style.maxWidth = '';
                    plot.style.maxHeight = '';
                    plot.style.margin = '';
                    plot.style.padding = '';
                    plot.style.position = '';
                    plot.style.top = '';
                    plot.style.left = '';
                    plot.style.display = '';
                    // Restore modebar
                    const modebar = plot.querySelector('.modebar');
                    if (modebar) modebar.style.display = 'block';
                });
            });
        });
    </script>
""", unsafe_allow_html=True)

# Table of Contents
toc_items = [
    ("Data Generation", "data-generation"),
    ("Summary Statistics", "summary-statistics"),
    ("Data Filtering", "data-filtering"),
    ("Publication Type Distribution by Year", "yearly-pub-distribution"),
    ("Publication Type Comparison (2023 vs 2024)", "pub-type-comparison"),
    ("Faculty/Department Quartile Distribution", "faculty-quartile"),
    ("Faculty and Publication Type Distribution", "faculty-pub-type"),
    ("Top 5 Units by Total Publications", "top-5-units"),
    ("Title-based Publication Distribution", "title-distribution"),
    ("Publication Change (2023-2024)", "pub-change"),
    ("Top 5 Researchers by Total Publications", "top-5-researchers"),
    ("Impact Score", "impact-score"),
    ("Publications and Impact Score", "pubs-impact"),
    ("Active Researcher Ratio", "active-ratio"),
    ("Publications by Year", "yearly-pubs"),
    ("Quartile Diversity Index", "diversity-index"),
    ("Most Common Publication Types", "common-pub-types"),
]

# Sidebar with TOC
with st.sidebar:
    st.header("Table of Contents")
    for title, anchor in toc_items:
        st.markdown(f'<a href="#{anchor}" style="text-decoration: none; color: #1f77b4;">{title}</a>', unsafe_allow_html=True)

# Faculty and Department mappings
FACULTY_MAPPING = {
    'Engineering Faculty': 'Engineering Faculty',
    'Faculty of Economics and Administrative Sciences': 'Faculty of Economics and Administrative Sciences',
    'Faculty of Arts and Sciences': 'Faculty of Arts and Sciences',
    'Faculty of Education': 'Faculty of Education',
    'Faculty of Architecture': 'Faculty of Architecture',
    'School of Foreign Languages': 'School of Foreign Languages',
    'Rectorate': 'Rectorate',
    'Institute of Natural Sciences': 'Institute of Natural Sciences',
    'Institute of Marine Sciences': 'Institute of Marine Sciences',
    'Institute of Applied Mathematics': 'Institute of Applied Mathematics',
    'Institute of Social Sciences': 'Institute of Social Sciences',
    'Institute of Informatics': 'Institute of Informatics',
    'Vocational School': 'Vocational School'
}
FACULTY_ORDER = sorted(FACULTY_MAPPING.keys())

DEPARTMENT_MAPPING = {
    'Actuarial Sciences': 'Actuarial Sciences',
    'Physical Education and Sports': 'Physical Education and Sports',
    'Computer Engineering': 'Computer Engineering',
    'Computer and Instructional Technologies Education': 'Computer and Instructional Technologies Education',
    'Science and Technology Policy Studies': 'Science and Technology Policy Studies',
    'Scientific Computing': 'Scientific Computing',
    'Information Systems': 'Information Systems',
    'Cognitive Sciences': 'Cognitive Sciences',
    'Biological Sciences': 'Biological Sciences',
    'Environmental Engineering': 'Environmental Engineering',
    'Marine Sciences': 'Marine Sciences',
    'Marine Biology and Fisheries': 'Marine Biology and Fisheries',
    'Marine Geology and Geophysics': 'Marine Geology and Geophysics',
    'Education Sciences': 'Education Sciences',
    'Electrical and Electronics Engineering': 'Electrical and Electronics Engineering',
    'Industrial Engineering': 'Industrial Engineering',
    'Industrial Design': 'Industrial Design',
    'Institute of Natural Sciences': 'Institute of Natural Sciences',
    'Philosophy': 'Philosophy',
    'Financial Mathematics': 'Financial Mathematics',
    'Physics': 'Physics',
    'Food Engineering': 'Food Engineering',
    'Aerospace Engineering': 'Aerospace Engineering',
    'Economics': 'Economics',
    'Civil Engineering': 'Civil Engineering',
    'Statistics': 'Statistics',
    'Geological Engineering': 'Geological Engineering',
    'Chemistry': 'Chemistry',
    'Chemical Engineering': 'Chemical Engineering',
    'Cryptography': 'Cryptography',
    'Mining Engineering': 'Mining Engineering',
    'Mechanical Engineering': 'Mechanical Engineering',
    'Mathematics': 'Mathematics',
    'Mathematics and Science Education': 'Mathematics and Science Education',
    'Vocational School': 'Vocational School',
    'Metallurgical and Materials Engineering': 'Metallurgical and Materials Engineering',
    'Architecture': 'Architecture',
    'Modeling and Simulation': 'Modeling and Simulation',
    'Modern Languages': 'Modern Languages',
    'Engineering Sciences': 'Engineering Sciences',
    'Music and Fine Arts': 'Music and Fine Arts',
    'Petroleum and Natural Gas Engineering': 'Petroleum and Natural Gas Engineering',
    'Psychology': 'Psychology',
    'Health Informatics': 'Health Informatics',
    'Cyber Security': 'Cyber Security',
    'Political Science and Public Administration': 'Political Science and Public Administration',
    'Sociology': 'Sociology',
    'Urban and Regional Planning': 'Urban and Regional Planning',
    'History': 'History',
    'Basic Education': 'Basic Education',
    'Turkish Language': 'Turkish Language',
    'International Relations': 'International Relations',
    'Data Informatics': 'Data Informatics',
    'Foreign Languages': 'Foreign Languages',
    'Foreign Language Education': 'Foreign Language Education',
    '': 'Unknown'
}
DEPARTMENT_ORDER = sorted([d for d in DEPARTMENT_MAPPING.keys() if d])

# Column definitions
COLUMNS = [
    'Title', 'Name', 'Faculty', 'Department', 'ESCI Articles', 'Scopus Articles',
    'Q1 Articles', 'Q2 Articles', 'Q3 Articles', 'Q4 Articles', 'Non-Quartile Articles', 'Year'
]

# Title order
title_order = ["Res. Asst.", "Lecturer", "Asst. Prof.", "Assoc. Prof.", "Prof.", "Other"]

# Publication types definition with fixed order
pub_types = ['ESCI Articles', 'Scopus Articles', 'Q1 Articles', 'Q2 Articles', 'Q3 Articles', 'Q4 Articles', 'Non-Quartile Articles']
pub_type_labels = {
    'ESCI Articles': 'ESCI Articles',
    'Scopus Articles': 'Scopus Articles',
    'Q1 Articles': 'Q1 Articles',
    'Q2 Articles': 'Q2 Articles',
    'Q3 Articles': 'Q3 Articles',
    'Q4 Articles': 'Q4 Articles',
    'Non-Quartile Articles': 'Non-Quartile Articles'
}

# Quartile columns with fixed order
quartile_cols = ['Q1 Articles', 'Q2 Articles', 'Q3 Articles', 'Q4 Articles']

# Function to generate random data
def generate_random_data(seed):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    np.random.seed(seed)
    num_researchers = np.random.randint(300, 500)
    logger.debug(f"Generating data for {num_researchers} researchers")
    
    data = []
    titles = ["Res. Asst.", "Lecturer", "Asst. Prof.", "Assoc. Prof.", "Prof.", "Other"]
    years = ['2023', '2024']
    
    faculty_to_depts = {
        'Engineering Faculty': ['Computer Engineering', 'Environmental Engineering', 'Electrical and Electronics Engineering', 'Industrial Engineering', 'Civil Engineering', 'Geological Engineering', 'Chemical Engineering', 'Mining Engineering', 'Mechanical Engineering', 'Metallurgical and Materials Engineering', 'Petroleum and Natural Gas Engineering', 'Aerospace Engineering', 'Food Engineering'],
        'Faculty of Economics and Administrative Sciences': ['Economics', 'Political Science and Public Administration', 'International Relations'],
        'Faculty of Arts and Sciences': ['Biological Sciences', 'Philosophy', 'Physics', 'Chemistry', 'Statistics', 'Mathematics', 'Psychology', 'Sociology', 'History'],
        'Faculty of Education': ['Computer and Instructional Technologies Education', 'Education Sciences', 'Mathematics and Science Education', 'Foreign Language Education', 'Basic Education'],
        'Faculty of Architecture': ['Architecture', 'Urban and Regional Planning', 'Industrial Design'],
        'School of Foreign Languages': ['Foreign Languages', 'Modern Languages'],
        'Rectorate': ['Turkish Language'],
        'Institute of Natural Sciences': ['Institute of Natural Sciences'],
        'Institute of Marine Sciences': ['Marine Sciences', 'Marine Biology and Fisheries', 'Marine Geology and Geophysics'],
        'Institute of Applied Mathematics': ['Financial Mathematics', 'Cryptography'],
        'Institute of Social Sciences': ['Science and Technology Policy Studies'],
        'Institute of Informatics': ['Information Systems', 'Cognitive Sciences', 'Scientific Computing', 'Health Informatics', 'Cyber Security', 'Data Informatics', 'Modeling and Simulation'],
        'Vocational School': ['Vocational School']
    }
    
    faculties = list(faculty_to_depts.keys())
    
    for i in range(num_researchers):
        logger.debug(f"Generating researcher {i+1}/{num_researchers}")
        title = np.random.choice(titles)
        name = f"Researcher_{np.random.randint(1000, 9999)}"
        faculty = np.random.choice(faculties)
        department = np.random.choice(faculty_to_depts[faculty])
        
        for year in years:
            esci = np.random.poisson(0.5)
            scopus = np.random.poisson(0.7)
            q1 = np.random.poisson(0.3)
            q2 = np.random.poisson(0.4)
            q3 = np.random.poisson(0.5)
            q4 = np.random.poisson(0.6)
            non_quartile = np.random.poisson(0.8)
            
            data.append({
                'Title': title,
                'Name': name,
                'Faculty': faculty,
                'Department': department,
                'ESCI Articles': esci,
                'Scopus Articles': scopus,
                'Q1 Articles': q1,
                'Q2 Articles': q2,
                'Q3 Articles': q3,
                'Q4 Articles': q4,
                'Non-Quartile Articles': non_quartile,
                'Year': year
            })
    
    df = pd.DataFrame(data)
    logger.debug(f"Generated DataFrame with shape: {df.shape}")
    return df

# Title extraction function
def extract_title(name):
    titles = [
        "Prof.", "Assoc. Prof.", "Asst. Prof.", "Lecturer", "Res. Asst.",
        "Research Assistant", "Instructor", "Researcher", "Administrative Staff"
    ]
    title_mapping = {
        "Prof.": "Prof.",
        "Assoc. Prof.": "Assoc. Prof.",
        "Asst. Prof.": "Asst. Prof.",
        "Lecturer": "Lecturer",
        "Res. Asst.": "Res. Asst.",
        "Research Assistant": "Res. Asst.",
        "Instructor": "Lecturer",
        "Researcher": "Other",
        "Administrative Staff": "Other"
    }
    if isinstance(name, str):
        for title in titles:
            if title in name:
                return title_mapping[title]
    return "Other"

# Error checking and removal function
def check_and_remove_errors(df):
    numeric_cols = ['ESCI Articles', 'Scopus Articles', 'Q1 Articles', 'Q2 Articles', 'Q3 Articles', 'Q4 Articles', 'Non-Quartile Articles']
    removed_entries = []
    neg_mask = df[numeric_cols].lt(0).any(axis=1)
    neg_indices = df[neg_mask].index
    for idx in neg_indices:
        researcher = df.loc[idx, "Name"]
        neg_cols = [col for col in numeric_cols if df.loc[idx, col] < 0]
        for col in neg_cols:
            reason = f"Negative value ({df.loc[idx, col]}) in {col} column"
            removed_entries.append({"Name": researcher, "Reason": reason})
    indices_to_remove = set(neg_indices)
    removed_df = pd.DataFrame(removed_entries)
    cleaned_df = df.drop(index=indices_to_remove).reset_index(drop=True)
    return cleaned_df, removed_df

# Data processing function
@st.cache_data
def load_and_process_data(seed):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug("Starting data generation")
        df = generate_random_data(seed)
        logger.debug("Data generation complete")
        
        df['Faculty'] = df['Faculty'].fillna('Unknown').astype(str)
        df['Department'] = df['Department'].fillna('Unknown').astype(str)
        df['Name'] = df['Name'].fillna('Unknown').astype(str)
        df = df.dropna(subset=['Name'], how='all')
        df = df.fillna({'ESCI Articles': 0, 'Scopus Articles': 0, 'Q1 Articles': 0,
                        'Q2 Articles': 0, 'Q3 Articles': 0, 'Q4 Articles': 0, 'Non-Quartile Articles': 0})
        df[['ESCI Articles', 'Scopus Articles', 'Q1 Articles', 'Q2 Articles', 'Q3 Articles',
            'Q4 Articles', 'Non-Quartile Articles']] = df[['ESCI Articles', 'Scopus Articles', 'Q1 Articles',
                                                          'Q2 Articles', 'Q3 Articles', 'Q4 Articles',
                                                          'Non-Quartile Articles']].astype(int)
        if 'Title' not in df.columns or df['Title'].isna().all():
            df['Title'] = df['Name'].apply(extract_title)
        else:
            df['Title'] = df['Title'].fillna('Other').astype(str)
        df['Total Publications'] = df[pub_types].sum(axis=1)
        df['Impact Score'] = (4 * df['Q1 Articles'] + 3 * df['Q2 Articles'] + 2 * df['Q3 Articles'] +
                             1 * df['Q4 Articles'] + 0.5 * (df['ESCI Articles'] + df['Scopus Articles'] +
                                                            df['Non-Quartile Articles']))
        df['Faculty'] = df['Faculty'].apply(lambda x: FACULTY_MAPPING.get(x, x))
        df['Department'] = df['Department'].apply(lambda x: DEPARTMENT_MAPPING.get(x, x))
        df['Faculty'] = pd.Categorical(df['Faculty'], categories=[FACULTY_MAPPING[f] for f in FACULTY_ORDER if FACULTY_MAPPING[f] in df['Faculty'].unique()], ordered=True)
        df['Department'] = pd.Categorical(df['Department'], categories=[DEPARTMENT_MAPPING[d] for d in DEPARTMENT_ORDER if DEPARTMENT_MAPPING[d] in df['Department'].unique()], ordered=True)
        df['Title'] = pd.Categorical(df['Title'], categories=title_order, ordered=True)
        df['Year'] = df['Year'].astype(str)
        df, removed_df = check_and_remove_errors(df)
        logger.debug("Data processing complete")
        return df, removed_df, None
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {str(e)}")
        return None, None, str(e)

# Color mapping
COLOR_MAP = {
    'ESCI Articles': '#1F77B4',
    'Scopus Articles': '#FF7F0E',
    'Q1 Articles': '#2CA02C',
    'Q2 Articles': '#D62728',
    'Q3 Articles': '#9467BD',
    'Q4 Articles': '#8C564B',
    'Non-Quartile Articles': '#7F7F7F'
}
YEAR_COLOR_MAP = {
    '2023': '#1F77B4',
    '2024': '#4A90E2'
}
CHANGE_COLOR_MAP = {
    'increase': '#2CA02C',
    'decrease': '#D62728'
}

# Generate and process data
with st.container():
    st.header("Data Generation", anchor="data-generation")
    st.markdown("This demo uses randomly generated publication data for 2023 and 2024 to simulate university research output. Originally, the app included a prompt to upload an Excel file.")
    
    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.removed_df = None
        st.session_state.error = None
        st.session_state.seed = np.random.randint(0, 1000000)
    
    if st.button("Scramble Data"):
        with st.spinner("Generating new synthetic data..."):
            st.session_state.seed = np.random.randint(0, 1000000)
            st.session_state.data, st.session_state.removed_df, st.session_state.error = load_and_process_data(st.session_state.seed)
    
    if st.session_state.data is None:
        with st.spinner("Generating synthetic data..."):
            st.session_state.data, st.session_state.removed_df, st.session_state.error = load_and_process_data(st.session_state.seed)
    
    df, removed_df, error = st.session_state.data, st.session_state.removed_df, st.session_state.error
    
    if error:
        st.error(f"Error processing data: {error}")
    else:
        st.success("Randomized data generated and faculty/department names standardized.")
        
        st.header("Summary Statistics", anchor="summary-statistics")
        st.markdown("This section summarizes key statistics and trends from the synthetic 2023 and 2024 publication data.")

        total_pubs = df['Total Publications'].sum()
        pubs_2023 = df[df['Year'] == '2023']['Total Publications'].sum()
        pubs_2024 = df[df['Year'] == '2024']['Total Publications'].sum()
        pub_change = ((pubs_2024 - pubs_2023) / pubs_2023 * 100) if pubs_2023 > 0 else 0
        active_researchers = df[df['Total Publications'] > 0]['Name'].nunique()
        high_impact_researchers = df[df['Q1 Articles'] + df['Q2 Articles'] > 0]['Name'].nunique()
        avg_impact = df['Impact Score'].mean()
        diversity_index = df.groupby(['Faculty', 'Year'])[quartile_cols].sum().apply(
            lambda row: -sum((c / sum(row) * np.log(c / sum(row))) for c in row if c > 0) if sum(row) > 0 else 0, axis=1
        ).mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Publications", f"{total_pubs}")
            st.metric("2023 Publications", f"{pubs_2023}")
            st.metric("2024 Publications", f"{pubs_2024}")
        with col2:
            st.metric("Publication Change (2023-2024)", f"{pub_change:.1f}%")
            st.metric("Active Researchers", f"{active_researchers}")
            st.metric("High-Impact Researchers (Q1/Q2)", f"{high_impact_researchers}")
        with col3:
            st.metric("Average Impact Score", f"{avg_impact:.2f}")
            st.metric("Average Quartile Diversity Index", f"{diversity_index:.2f}")

        st.subheader("Key Insights")
        takeaways = []
        if pub_change > 10:
            takeaways.append(f"- **Growth Trend**: A {pub_change:.1f}% increase in publications from 2023 to 2024 suggests a strong rise in research output (synthetic data).")
        elif pub_change < -10:
            takeaways.append(f"- **Decline Trend**: A {pub_change:.1f}% decrease in publications from 2023 to 2024 may indicate shifts in resources or focus (synthetic data).")
        top_pub_type = df[pub_types].sum().idxmax()
        top_pub_count = df[pub_types].sum().max()
        takeaways.append(f"- **Dominant Publication Type**: {pub_type_labels[top_pub_type]} ({top_pub_count} articles) constitutes a significant portion of output (synthetic data).")
        if high_impact_researchers / active_researchers > 0.5:
            takeaways.append(f"- **High-Impact Focus**: {high_impact_researchers/active_researchers*100:.1f}% of active researchers produce Q1 or Q2 articles, indicating a strong emphasis on high-impact research (synthetic data).")
        declining_faculties = df.groupby(['Faculty', 'Year'])['Total Publications'].sum().unstack().fillna(0).pipe(
            lambda x: x[x['2024'] < x['2023'] * 0.8].index.tolist()
        )
        if declining_faculties:
            takeaways.append(f"- **Declining Faculties**: {', '.join(declining_faculties)} experienced significant publication drops in 2024 (synthetic data).")
        for takeaway in takeaways:
            st.markdown(takeaway)

        st.subheader("Trend Visualizations")
        with st.expander("View Trends"):
            tab1, tab2, tab3 = st.tabs(["Publication Type Changes", "Publication Type Distribution", "Most Active Faculties"])
            with tab1:
                pub_trend = df.groupby('Year')[pub_types].sum().reset_index()
                pub_trend_melted = pub_trend.melt(id_vars='Year', value_vars=pub_types, var_name='Publication Type', value_name='Count')
                pub_trend_melted['Publication Type'] = pd.Categorical(
                    pub_trend_melted['Publication Type'].map(pub_type_labels),
                    categories=[pub_type_labels[pt] for pt in pub_types],
                    ordered=True
                )
                change_data = pub_trend.set_index('Year')[pub_types].T
                change_data['Change'] = change_data['2024'] - change_data['2023']
                change_data['Change Type'] = change_data['Change'].apply(lambda x: 'increase' if x >= 0 else 'decrease')
                change_data = change_data.reset_index().rename(columns={'index': 'Publication Type'})
                change_data['Publication Type'] = pd.Categorical(
                    change_data['Publication Type'].map(pub_type_labels),
                    categories=[pub_type_labels[pt] for pt in pub_types],
                    ordered=True
                )
                fig_trend = px.bar(change_data, x='Change', y='Publication Type', orientation='h',
                                   color='Change Type', title="Publication Type Changes (2023-2024)",
                                   color_discrete_map=CHANGE_COLOR_MAP)
                fig_trend.update_layout(
                    xaxis_title="Change (2024 - 2023)",
                    yaxis_title="Publication Type",
                    height=500,
                    width=900,
                    xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2)
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="pub_trend_change_bar")
            with tab2:
                pub_dist = df[pub_types].sum().reset_index()
                pub_dist.columns = ['Publication Type', 'Count']
                pub_dist['Publication Type'] = pd.Categorical(
                    pub_dist['Publication Type'].map(pub_type_labels),
                    categories=[pub_type_labels[pt] for pt in pub_types],
                    ordered=True
                )
                fig_dist = px.pie(pub_dist, names='Publication Type', values='Count',
                                  title="Distribution of All Publication Types", color_discrete_map=COLOR_MAP)
                fig_dist.update_layout(height=500, width=900)
                st.plotly_chart(fig_dist, use_container_width=True, key="pub_dist_pie_summary")
            with tab3:
                faculty_pubs = df.groupby(['Faculty', 'Year'])['Total Publications'].sum().reset_index()
                faculty_pubs['Faculty'] = faculty_pubs['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
                top_faculties = faculty_pubs.groupby('Faculty')['Total Publications'].sum().nlargest(5).index
                top_faculty_data = faculty_pubs[faculty_pubs['Faculty'].isin(top_faculties)]
                fig_faculty = px.bar(top_faculty_data, x='Faculty', y='Total Publications', color='Year', barmode='group',
                                     title="Top 5 Most Active Faculties (Publication Count)", color_discrete_map=YEAR_COLOR_MAP)
                fig_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
                st.plotly_chart(fig_faculty, use_container_width=True, key="top_faculties_bar_summary")

        unmapped_faculties = df['Faculty'][~df['Faculty'].isin(FACULTY_MAPPING.values()) & (df['Faculty'] != 'Unknown')].unique()
        unmapped_departments = df['Department'][~df['Department'].isin(DEPARTMENT_MAPPING.values()) & (df['Department'] != 'Unknown')].unique()
        if unmapped_faculties.size > 0 or unmapped_departments.size > 0:
            with st.expander("Unmapped Faculties and Departments"):
                if unmapped_faculties.size > 0:
                    st.warning("Unmapped Faculties:")
                    for faculty in unmapped_faculties:
                        st.write(f"- {faculty}")
                if unmapped_departments.size > 0:
                    st.warning("Unmapped Departments:")
                    for dept in unmapped_departments:
                        st.write(f"- {dept}")

        st.header("Data Filtering", anchor="data-filtering")
        st.markdown("Filter data by faculty, department, and title. Select categories to include or exclude.")
        col1, col2, col3 = st.columns(3)
        with col1:
            faculty_options = sorted([f for f in FACULTY_MAPPING.keys() if f in df['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x).unique()], key=str.lower)
            faculty_filter = st.multiselect("Select Faculty (Include)", options=faculty_options, default=[])
        with col2:
            dept_options = sorted([d for d in DEPARTMENT_MAPPING.keys() if d in df['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x).unique() and d], key=str.lower)
            department_filter = st.multiselect("Select Department (Include)", options=dept_options, default=[])
        with col3:
            title_options = sorted([t for t in df['Title'].unique() if t and isinstance(t, str)], key=str.lower)
            title_filter = st.multiselect("Select Title (Include)", options=title_options, default=[])
        
        col4, col5, col6 = st.columns(3)
        with col4:
            faculty_exclude = st.multiselect("Exclude Faculty", options=faculty_options, default=[])
        with col5:
            department_exclude = st.multiselect("Exclude Department", options=dept_options, default=[])
        with col6:
            title_exclude = st.multiselect("Exclude Title", options=title_options, default=[])

        filtered_df = df.copy()
        if faculty_filter:
            filtered_df = filtered_df[filtered_df['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x).isin(faculty_filter)]
        if department_filter:
            filtered_df = filtered_df[filtered_df['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x).isin(department_filter)]
        if title_filter:
            filtered_df = filtered_df[filtered_df['Title'].isin(title_filter)]
        if faculty_exclude:
            filtered_df = filtered_df[~filtered_df['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x).isin(faculty_exclude)]
        if department_exclude:
            filtered_df = filtered_df[~filtered_df['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x).isin(department_exclude)]
        if title_exclude:
            filtered_df = filtered_df[~filtered_df['Title'].isin(title_exclude)]

        st.subheader("Data Visualizations")

        st.header("Publication Type Distribution by Year", anchor="yearly-pub-distribution")
        st.markdown("Distribution of publication types for 2023 and 2024.")
        pub_data = filtered_df.groupby('Year')[pub_types].sum().reset_index()
        pub_data_melted = pub_data.melt(id_vars='Year', value_vars=pub_types, var_name='Publication Type', value_name='Count')
        pub_data_melted['Publication Type'] = pd.Categorical(
            pub_data_melted['Publication Type'].map(pub_type_labels),
            categories=[pub_type_labels[pt] for pt in pub_types],
            ordered=True
        )
        fig1 = px.bar(pub_data_melted, x='Year', y='Count', color='Publication Type', barmode='group',
                      title="Publication Type Distribution by Year", color_discrete_map=COLOR_MAP,
                      category_orders={'Publication Type': [pub_type_labels[pt] for pt in pub_types]})
        fig1.update_layout(height=600, width=900)
        st.plotly_chart(fig1, use_container_width=True, key="year_wise_pub_type")

        st.header("Publication Type Comparison (2023 vs 2024)", anchor="pub-type-comparison")
        st.markdown("Comparison of publication types between 2023 and 2024.")
        pub_compare = filtered_df.groupby(['Year'])[pub_types].sum().reset_index()
        pub_compare_melted = pub_compare.melt(id_vars='Year', value_vars=pub_types, var_name='Publication Type', value_name='Count')
        pub_compare_melted['Publication Type'] = pd.Categorical(
            pub_compare_melted['Publication Type'].map(pub_type_labels),
            categories=[pub_type_labels[pt] for pt in pub_types],
            ordered=True
        )
        fig1b = px.bar(pub_compare_melted, x='Publication Type', y='Count', color='Year', barmode='group',
                       title="Publication Type Comparison (2023 vs 2024)",
                       color_discrete_map=YEAR_COLOR_MAP,
                       category_orders={'Publication Type': [pub_type_labels[pt] for pt in pub_types]})
        fig1b.update_layout(xaxis_tickangle=45, height=600, width=900)
        st.plotly_chart(fig1b, use_container_width=True, key="year_comparison_bar")

        st.header("Faculty/Department Quartile Distribution", anchor="faculty-quartile")
        st.markdown("Distribution of Q1-Q4 quartile articles across faculties and departments.")
        with st.expander("View Faculty/Department Quartile Distributions"):
            tab1, tab2 = st.tabs(["Faculties", "Departments"])
            with tab1:
                pub_quartile_data_faculty = filtered_df.groupby(['Faculty', 'Year'])[quartile_cols].sum().reset_index()
                pub_quartile_data_faculty['Faculty'] = pub_quartile_data_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
                pub_quartile_melted_faculty = pub_quartile_data_faculty.melt(id_vars=['Faculty', 'Year'], value_vars=quartile_cols, var_name='Quartile', value_name='Count')
                pub_quartile_melted_faculty['Quartile'] = pd.Categorical(
                    pub_quartile_melted_faculty['Quartile'],
                    categories=quartile_cols,
                    ordered=True
                )
                for year in ['2023', '2024']:
                    for faculty in pub_quartile_data_faculty['Faculty'].unique():
                        faculty_data = pub_quartile_melted_faculty[(pub_quartile_melted_faculty['Faculty'] == faculty) & (pub_quartile_melted_faculty['Year'] == year)]
                        if faculty_data['Count'].sum() > 0:
                            fig = px.pie(faculty_data, names='Quartile', values='Count', title=f"{year} - {faculty} Quartile Distribution",
                                         color_discrete_map=COLOR_MAP,
                                         category_orders={'Quartile': quartile_cols})
                            fig.update_layout(height=500, width=800)
                            st.plotly_chart(fig, use_container_width=True, key=f"pie_faculty_{faculty}_{year}")
            with tab2:
                pub_quartile_data_dept = filtered_df.groupby(['Department', 'Year'])[quartile_cols].sum().reset_index()
                pub_quartile_data_dept['Department'] = pub_quartile_data_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
                pub_quartile_melted_dept = pub_quartile_data_dept.melt(id_vars=['Department', 'Year'], value_vars=quartile_cols, var_name='Quartile', value_name='Count')
                pub_quartile_melted_dept['Quartile'] = pd.Categorical(
                    pub_quartile_melted_dept['Quartile'],
                    categories=quartile_cols,
                    ordered=True
                )
                for year in ['2023', '2024']:
                    for dept in pub_quartile_data_dept['Department'].unique():
                        dept_data = pub_quartile_melted_dept[(pub_quartile_melted_dept['Department'] == dept) & (pub_quartile_melted_dept['Year'] == year)]
                        if dept_data['Count'].sum() > 0:
                            fig = px.pie(dept_data, names='Quartile', values='Count', title=f"{year} - {dept} Quartile Distribution",
                                         color_discrete_map=COLOR_MAP,
                                         category_orders={'Quartile': quartile_cols})
                            fig.update_layout(height=500, width=800)
                            st.plotly_chart(fig, use_container_width=True, key=f"pie_dept_{dept}_{year}")

        st.header("Faculty and Publication Type Distribution", anchor="faculty-pub-type")
        st.markdown("Heatmap of publication types across faculties.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            heatmap_data_faculty = filtered_df.groupby(['Faculty', 'Year'])[pub_types].sum().reset_index()
            heatmap_data_faculty['Faculty'] = heatmap_data_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            for year in ['2023', '2024']:
                heatmap_year = heatmap_data_faculty[heatmap_data_faculty['Year'] == year].set_index('Faculty')[pub_types]
                heatmap_year = heatmap_year.loc[heatmap_year.sum(axis=1) > 0, :]
                heatmap_year = heatmap_year.reindex(index=[FACULTY_MAPPING[f] for f in FACULTY_ORDER if FACULTY_MAPPING[f] in heatmap_year.index])
                heatmap_year = heatmap_year[pub_types]  # Ensure column order
                heatmap_year.columns = [pub_type_labels[col] for col in heatmap_year.columns]
                fig2 = px.imshow(heatmap_year, title=f"{year} - Faculty and Publication Type Distribution",
                                 labels=dict(x="Publication Type", y="Faculty", color="Count"), color_continuous_scale="Blues")
                fig2.update_layout(height=800, width=900, xaxis={'tickangle': 45})
                st.plotly_chart(fig2, use_container_width=True, key=f"heatmap_faculty_{year}")
        with tab2:
            heatmap_data_dept = filtered_df.groupby(['Department', 'Year'])[pub_types].sum().reset_index()
            heatmap_data_dept['Department'] = heatmap_data_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            for year in ['2023', '2024']:
                heatmap_year = heatmap_data_dept[heatmap_data_dept['Year'] == year].set_index('Department')[pub_types]
                heatmap_year = heatmap_year.loc[heatmap_year.sum(axis=1) > 0, :]
                heatmap_year = heatmap_year.reindex(index=[DEPARTMENT_MAPPING[d] for d in DEPARTMENT_ORDER if DEPARTMENT_MAPPING[d] in heatmap_year.index])
                heatmap_year = heatmap_year[pub_types]  # Ensure column order
                heatmap_year.columns = [pub_type_labels[col] for col in heatmap_year.columns]
                fig2 = px.imshow(heatmap_year, title=f"{year} - Department and Publication Type Distribution",
                                 labels=dict(x="Publication Type", y="Department", color="Count"), color_continuous_scale="Blues")
                fig2.update_layout(height=800, width=900, xaxis={'tickangle': 45})
                st.plotly_chart(fig2, use_container_width=True, key=f"heatmap_dept_{year}")

        st.header("Top 5 Units by Total Publications", anchor="top-5-units")
        st.markdown("Top 5 faculties and departments by publication count for 2023 and 2024.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            faculty_year_totals = filtered_df.groupby(['Faculty', 'Year'])['Total Publications'].sum().reset_index()
            faculty_year_totals['Faculty'] = faculty_year_totals['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FAC pleasuresY_MAPPING.values() else x)
            top_5_faculties = faculty_year_totals.groupby('Faculty')['Total Publications'].sum().nlargest(5).index
            top_faculty_data = faculty_year_totals[faculty_year_totals['Faculty'].isin(top_5_faculties)]
            fig3_faculty = px.bar(top_faculty_data, x='Faculty', y='Total Publications', color='Year', barmode='group',
                                  title="Top 5 Faculties by Total Publications",
                                  color_discrete_map=YEAR_COLOR_MAP)
            fig3_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig3_faculty, use_container_width=True, key="top_5_faculties_bar")
        with tab2:
            dept_year_totals = filtered_df.groupby(['Department', 'Year'])['Total Publications'].sum().reset_index()
            dept_year_totals['Department'] = dept_year_totals['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            top_5_depts = dept_year_totals.groupby('Department')['Total Publications'].sum().nlargest(5).index
            top_dept_data = dept_year_totals[dept_year_totals['Department'].isin(top_5_depts)]
            fig3_dept = px.bar(top_dept_data, x='Department', y='Total Publications', color='Year', barmode='group',
                               title="Top 5 Departments by Total Publications",
                               color_discrete_map=YEAR_COLOR_MAP)
            fig3_dept.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig3_dept, use_container_width=True, key="top_5_depts_bar")
        
        st.header("Title-based Publication Distribution", anchor="title-distribution")
        st.markdown("Publication contributions by academic titles for 2023 and 2024.")
        
        title_mapping = {
            "Prof.": "Prof.",
            "Assoc. Prof.": "Assoc. Prof.",
            "Asst. Prof.": "Asst. Prof.",
            "Research Assistant": "Other Titles",
            "Instructor": "Other Titles",
            "Researcher": "Other Titles",
            "Administrative Staff": "Other Titles",
            "Other": "Other Titles",
            "Res. Asst.": "Other Titles",
            "Lecturer": "Other Titles"
        }
        filtered_df['Title'] = filtered_df['Title'].map(title_mapping).fillna("Other Titles")
        title_year_pubs = filtered_df.groupby(['Title', 'Year'])['Total Publications'].sum().reset_index()
        title_order_updated = ["Asst. Prof.", "Assoc. Prof.", "Prof.", "Other Titles"]
        title_year_pubs['Title'] = pd.Categorical(title_year_pubs['Title'], categories=title_order_updated, ordered=True)
        
        st.write("**Publication Counts by Title**")
        st.dataframe(title_year_pubs.pivot(index='Title', columns='Year', values='Total Publications').fillna(0))
        
        fig_titles = px.bar(title_year_pubs, x='Title', y='Total Publications', color='Year', barmode='group',
                            title="Publication Distribution by Title",
                            color_discrete_map=YEAR_COLOR_MAP)
        max_y = title_year_pubs['Total Publications'].max() + 10 if not title_year_pubs.empty else 100
        fig_titles.update_layout(
            xaxis_tickangle=45,
            height=500,
            width=900,
            xaxis_title="Title",
            yaxis=dict(range=[0, max_y], title="Total Publications")
        )
        st.plotly_chart(fig_titles, use_container_width=True, key="title_wise_pubs_bar")
        
        st.header("Publication Change (2023-2024)", anchor="pub-change")
        st.markdown("Change in publication counts across units from 2023 to 2024.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            faculty_year_pubs = filtered_df.groupby(['Faculty', 'Year'])['Total Publications'].sum().unstack().fillna(0)
            faculty_year_pubs['Change'] = faculty_year_pubs['2024'] - faculty_year_pubs['2023']
            faculty_year_pubs['Change Type'] = faculty_year_pubs['Change'].apply(lambda x: 'increase' if x >= 0 else 'decrease')
            change_df_faculty = faculty_year_pubs.reset_index()[['Faculty', 'Change', 'Change Type']]
            change_df_faculty['Faculty'] = change_df_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            fig6_faculty = px.bar(change_df_faculty, x='Faculty', y='Change', color='Change Type',
                                  title="Faculty Publication Change (2023-2024)",
                                  color_discrete_map=CHANGE_COLOR_MAP)
            fig6_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig6_faculty, use_container_width=True, key="pub_change_faculty_bar")
        with tab2:
            dept_year_pubs = filtered_df.groupby(['Department', 'Year'])['Total Publications'].sum().unstack().fillna(0)
            dept_year_pubs['Change'] = dept_year_pubs['2024'] - dept_year_pubs['2023']
            dept_year_pubs['Change Type'] = dept_year_pubs['Change'].apply(lambda x: 'increase' if x >= 0 else 'decrease')
            change_df_dept = dept_year_pubs.reset_index()[['Department', 'Change', 'Change Type']]
            change_df_dept['Department'] = change_df_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            fig6_dept = px.bar(change_df_dept, x='Department', y='Change', color='Change Type',
                               title="Department Publication Change (2023-2024)",
                               color_discrete_map=CHANGE_COLOR_MAP)
            fig6_dept.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig6_dept, use_container_width=True, key="pub_change_dept_bar")

        st.header("Top 5 Researchers by Total Publications", anchor="top-5-researchers")
        st.markdown("Top 5 most active researchers.")
        top_researchers = filtered_df.groupby(['Name', 'Faculty', 'Department'])['Total Publications'].sum().reset_index()
        top_researchers['Faculty'] = top_researchers['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
        top_researchers['Department'] = top_researchers['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
        top_5_researchers = top_researchers.nlargest(5, 'Total Publications')
        fig7 = go.Figure(data=[
            go.Table(
                header=dict(values=['Name', 'Faculty', 'Department', 'Total Publications'],
                            fill_color='paleturquoise', align='left'),
                cells=dict(values=[top_5_researchers['Name'], top_5_researchers['Faculty'],
                                  top_5_researchers['Department'], top_5_researchers['Total Publications']],
                           fill_color='lavender', align='left'))
        ])
        st.plotly_chart(fig7, use_container_width=True, key="top_5_researchers_table")

        st.header("Impact Score", anchor="impact-score")
        st.markdown("Total impact score across units for 2023 and 2024.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            faculty_year_impact = filtered_df.groupby(['Faculty', 'Year'])['Impact Score'].sum().reset_index()
            faculty_year_impact['Faculty'] = faculty_year_impact['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            fig8_faculty = px.bar(faculty_year_impact, x='Faculty', y='Impact Score', color='Year', barmode='group',
                                  title="Faculty Impact Score",
                                  color_discrete_map=YEAR_COLOR_MAP)
            fig8_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig8_faculty, use_container_width=True, key="impact_score_faculty_bar")
        with tab2:
            dept_year_impact = filtered_df.groupby(['Department', 'Year'])['Impact Score'].sum().reset_index()
            dept_year_impact['Department'] = dept_year_impact['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            fig8_dept = px.bar(dept_year_impact, x='Department', y='Impact Score', color='Year', barmode='group',
                               title="Department Impact Score",
                               color_discrete_map=YEAR_COLOR_MAP)
            fig8_dept.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig8_dept, use_container_width=True, key="impact_score_dept_bar")
        st.markdown("**Calculation:** The impact score is a weighted sum based on publication types: Q1 articles (4 points), Q2 (3 points), Q3 (2 points), Q4 (1 point), and ESCI, Scopus, and non-quartile articles (0.5 points each). Formula: **Impact Score = (Q1 × 4) + (Q2 × 3) + (Q3 × 2) + (Q4 × 1) + [(ESCI + Scopus + Non-Quartile) × 0.5]**. Example: A faculty with 5 Q1 (5×4=20), 3 Q2 (3×3=9), and 2 ESCI (2×0.5=1) articles has a total impact score of 20+9+1=30.")

        st.header("Publications and Impact Score", anchor="pubs-impact")
        st.markdown("Relationship between total publications and impact score across units.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            scatter_data_faculty = filtered_df.groupby(['Faculty', 'Year'])[['Total Publications', 'Impact Score']].sum().reset_index()
            scatter_data_faculty['Faculty'] = scatter_data_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            fig8b_faculty = px.scatter(scatter_data_faculty, x='Total Publications', y='Impact Score', color='Faculty', symbol='Year',
                                       title="Faculty Publications and Impact Score", size='Total Publications')
            fig8b_faculty.update_layout(height=600, width=900)
            st.plotly_chart(fig8b_faculty, use_container_width=True, key="scatter_faculty_plot")
        with tab2:
            scatter_data_dept = filtered_df.groupby(['Department', "Year"])[['Total Publications', 'Impact Score']].sum().reset_index()
            scatter_data_dept['Department'] = scatter_data_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            fig8b_dept = px.scatter(scatter_data_dept, x='Total Publications', y='Impact Score', color='Department', symbol='Year',
                                    title="Department Publications and Impact Score", size='Total Publications')
            fig8b_dept.update_layout(height=600, width=900)
            st.plotly_chart(fig8b_dept, use_container_width=True, key="scatter_dept_plot")
        st.markdown("**Calculation:** This visualization compares total publications (sum of all publication types) and impact scores (calculated as above). Each point represents a faculty or department, with point size indicating total publications and symbols indicating the year (2023 or 2024).")

        st.header("Active Researcher Ratio", anchor="active-ratio")
        st.markdown("Proportion of active researchers across units.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            total_researchers_faculty = filtered_df.groupby(['Faculty', 'Year'])['Name'].nunique().reset_index(name='Total Researchers')
            active_researchers_faculty = filtered_df[filtered_df['Total Publications'] > 0].groupby(['Faculty', 'Year'])['Name'].nunique().reset_index(name='Active Researchers')
            total_researchers_faculty['Faculty'] = total_researchers_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            active_researchers_faculty['Faculty'] = active_researchers_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            ratio_df_faculty = total_researchers_faculty.merge(active_researchers_faculty, on=['Faculty', 'Year'], how='left').fillna({'Active Researchers': 0})
            ratio_df_faculty['Active Researcher Ratio'] = ratio_df_faculty['Active Researchers'] / ratio_df_faculty['Total Researchers']
            fig9_faculty = px.bar(ratio_df_faculty, x='Faculty', y='Active Researcher Ratio', color='Year', barmode='group',
                                  title="Faculty Active Researcher Ratio",
                                  color_discrete_map=YEAR_COLOR_MAP)
            fig9_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig9_faculty, use_container_width=True, key="active_ratio_faculty_bar")
        with tab2:
            total_researchers_dept = filtered_df.groupby(['Department', 'Year'])['Name'].nunique().reset_index(name='Total Researchers')
            active_researchers_dept = filtered_df[filtered_df['Total Publications'] > 0].groupby(['Department', 'Year'])['Name'].nunique().reset_index(name='Active Researchers')
            total_researchers_dept['Department'] = total_researchers_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            active_researchers_dept['Department'] = active_researchers_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            ratio_df_dept = total_researchers_dept.merge(active_researchers_dept, on=['Department', 'Year'], how='left').fillna({'Active Researchers': 0})
            ratio_df_dept['Active Researcher Ratio'] = ratio_df_dept['Active Researchers'] / ratio_df_dept['Total Researchers']
            fig9_dept = px.bar(ratio_df_dept, x='Department', y='Active Researcher Ratio', color='Year', barmode='group',
                               title="Department Active Researcher Ratio",
                               color_discrete_map=YEAR_COLOR_MAP)
            fig9_dept.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig9_dept, use_container_width=True, key="active_ratio_dept_bar")

        st.header("Publications by Year", anchor="yearly-pubs")
        st.markdown("Publication counts across units for 2023 and 2024.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            faculty_year_pubs = filtered_df.groupby(['Faculty', 'Year'])['Total Publications'].sum().reset_index()
            faculty_year_pubs['Faculty'] = faculty_year_pubs['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            fig10_faculty = px.bar(faculty_year_pubs, x='Faculty', y='Total Publications', color='Year', barmode='group',
                                   title="Faculty Publications by Year",
                                   color_discrete_map=YEAR_COLOR_MAP)
            fig10_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig10_faculty, use_container_width=True, key="pubs_by_year_faculty_bar")
        with tab2:
            dept_year_pubs = filtered_df.groupby(['Department', 'Year'])['Total Publications'].sum().reset_index()
            dept_year_pubs['Department'] = dept_year_pubs['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            fig10_dept = px.bar(dept_year_pubs, x='Department', y='Total Publications', color='Year', barmode='group',
                                title="Department Publications by Year",
                                color_discrete_map=YEAR_COLOR_MAP)
            fig10_dept.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig10_dept, use_container_width=True, key="pubs_by_year_dept_bar")

        st.header("Quartile Diversity Index", anchor="diversity-index")
        st.markdown("Diversity of Q1-Q4 articles across units.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            def shannon_diversity(row):
                counts = [row[col] for col in quartile_cols]
                total = sum(counts)
                if total == 0:
                    return 0
                proportions = [c / total for c in counts if c > 0]
                return -sum(p * np.log(p) for p in proportions) if proportions else 0
            faculty_quartile = filtered_df.groupby(['Faculty', 'Year'])[quartile_cols].sum().reset_index()
            faculty_quartile['Diversity Index'] = faculty_quartile[quartile_cols].apply(shannon_diversity, axis=1)
            faculty_quartile['Faculty'] = faculty_quartile['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            fig11_faculty = px.bar(faculty_quartile, x='Faculty', y='Diversity Index', color='Year', barmode='group',
                                   title="Faculty Quartile Diversity Index",
                                   color_discrete_map=YEAR_COLOR_MAP)
            fig11_faculty.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig11_faculty, use_container_width=True, key="diversity_index_faculty_bar")
        with tab2:
            dept_quartile = filtered_df.groupby(['Department', 'Year'])[quartile_cols].sum().reset_index()
            dept_quartile['Diversity Index'] = dept_quartile[quartile_cols].apply(shannon_diversity, axis=1)
            dept_quartile['Department'] = dept_quartile['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            fig11_dept = px.bar(dept_quartile, x='Department', y='Diversity Index', color='Year', barmode='group',
                                title="Department Quartile Diversity Index",
                                color_discrete_map=YEAR_COLOR_MAP)
            fig11_dept.update_layout(xaxis_tickangle=45, height=500, width=900)
            st.plotly_chart(fig11_dept, use_container_width=True, key="diversity_index_dept_bar")
        st.markdown("**Calculation:** The Quartile Diversity Index is calculated using the Shannon Entropy formula, considering the proportions of Q1, Q2, Q3, and Q4 articles. Formula: **H = -Σ(p_i * ln(p_i))**; where p_i is the proportion of each quartile type. Example: A faculty with 10 Q1, 5 Q2, 5 Q3, and 0 Q4 articles (total 20) has proportions Q1=0.5, Q2=0.25, Q3=0.25. H = -[(0.5 * ln(0.5)) + (0.25 * ln(0.25)) + (0.25 * ln(0.25))] ≈ 1.04. Higher indices indicate a more balanced quartile distribution.")

        st.header("Most Common Publication Types", anchor="common-pub-types")
        st.markdown("Most common publication types across units.")
        tab1, tab2 = st.tabs(["Faculties", "Departments"])
        with tab1:
            pub_type_sums_faculty = filtered_df.groupby(['Faculty', 'Year'])[pub_types].sum().reset_index()
            pub_type_sums_faculty['Faculty'] = pub_type_sums_faculty['Faculty'].map(lambda x: list(FACULTY_MAPPING.keys())[list(FACULTY_MAPPING.values()).index(x)] if x in FACULTY_MAPPING.values() else x)
            pub_type_melted_faculty = pub_type_sums_faculty.melt(id_vars=['Faculty', 'Year'], value_vars=pub_types, var_name='Publication Type', value_name='Count')
            pub_type_melted_faculty['Publication Type'] = pd.Categorical(
                pub_type_melted_faculty['Publication Type'].map(pub_type_labels),
                categories=[pub_type_labels[pt] for pt in pub_types],
                ordered=True
            )
            
            # 2023 Chart
            faculty_2023 = pub_type_melted_faculty[pub_type_melted_faculty['Year'] == '2023']
            fig12_faculty_2023 = px.bar(faculty_2023, x='Faculty', y='Count', color='Publication Type', barmode='stack',
                                        title="Most Common Publication Types by Faculty (2023)", color_discrete_map=COLOR_MAP,
                                        category_orders={'Publication Type': [pub_type_labels[pt] for pt in pub_types]})
            fig12_faculty_2023.update_layout(xaxis_tickangle=45, height=600, width=900)
            st.plotly_chart(fig12_faculty_2023, use_container_width=True, key="faculty_top_pub_types_2023")
            
            # 2024 Chart
            faculty_2024 = pub_type_melted_faculty[pub_type_melted_faculty['Year'] == '2024']
            fig12_faculty_2024 = px.bar(faculty_2024, x='Faculty', y='Count', color='Publication Type', barmode='stack',
                                        title="Most Common Publication Types by Faculty (2024)", color_discrete_map=COLOR_MAP,
                                        category_orders={'Publication Type': [pub_type_labels[pt] for pt in pub_types]})
            fig12_faculty_2024.update_layout(xaxis_tickangle=45, height=600, width=900)
            st.plotly_chart(fig12_faculty_2024, use_container_width=True, key="faculty_top_pub_types_2024")
            
        with tab2:
            pub_type_sums_dept = filtered_df.groupby(['Department', 'Year'])[pub_types].sum().reset_index()
            pub_type_sums_dept['Department'] = pub_type_sums_dept['Department'].map(lambda x: list(DEPARTMENT_MAPPING.keys())[list(DEPARTMENT_MAPPING.values()).index(x)] if x in DEPARTMENT_MAPPING.values() else x)
            pub_type_melted_dept = pub_type_sums_dept.melt(id_vars=['Department', 'Year'], value_vars=pub_types, var_name='Publication Type', value_name='Count')
            pub_type_melted_dept['Publication Type'] = pd.Categorical(
                pub_type_melted_dept['Publication Type'].map(pub_type_labels),
                categories=[pub_type_labels[pt] for pt in pub_types],
                ordered=True
            )
            
            # 2023 Chart
            dept_2023 = pub_type_melted_dept[pub_type_melted_dept['Year'] == '2023']
            fig12_dept_2023 = px.bar(dept_2023, x='Department', y='Count', color='Publication Type', barmode='stack',
                                     title="Most Common Publication Types by Department (2023)", color_discrete_map=COLOR_MAP,
                                     category_orders={'Publication Type': [pub_type_labels[pt] for pt in pub_types]})
            fig12_dept_2023.update_layout(xaxis_tickangle=45, height=600, width=900)
            st.plotly_chart(fig12_dept_2023, use_container_width=True, key="dept_top_pub_types_2023")
            
            # 2024 Chart
            dept_2024 = pub_type_melted_dept[pub_type_melted_dept['Year'] == '2024']
            fig12_dept_2024 = px.bar(dept_2024, x='Department', y='Count', color='Publication Type', barmode='stack',
                                     title="Most Common Publication Types by Department (2024)", color_discrete_map=COLOR_MAP,
                                     category_orders={'Publication Type': [pub_type_labels[pt] for pt in pub_types]})
            fig12_dept_2024.update_layout(xaxis_tickangle=45, height=600, width=900)
            st.plotly_chart(fig12_dept_2024, use_container_width=True, key="dept_top_pub_types_2024")

st.markdown("---")
st.markdown("Kerem Delialioğlu")
st.markdown("Middle East Technical University — Office of Research Coordination")
st.markdown("2025")
