# visibility_dashboard.py
import re, collections
from pathlib import Path
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ═════════════ 0) Layout & Colours ═════════════
st.set_page_config("AI Visibility Monitor", layout="wide")
st.title("AI Visibility Monitor")

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
def color_map(keys):
    pal = (_PALETTE * (len(keys)//len(_PALETTE)+1))[:len(keys)]
    return dict(zip(keys, pal))


# ═════════════ 1) CSV import ═════════════════
def read_all_csv(folder="data"):
    files = list(Path(folder).glob("*.csv"))
    if not files:
        st.error("No CSV files in ./data"); st.stop()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = read_all_csv()

def parse_date(s:pd.Series)->pd.Series:
    txt = s.astype(str).str.strip().str.replace(",",".",regex=False)
    d1  = pd.to_datetime(txt, format="%d.%m.%Y", errors="coerce")
    d2  = pd.to_datetime(txt, errors="coerce", dayfirst=True)
    return d1.fillna(d2).dt.normalize()

df["Date"] = parse_date(df["Date"])
if df["Date"].isna().all():
    st.error("No valid dates parsed."); st.stop()

for c in ["intent","category","route"]:
    if c in df:
        df[c] = df[c].astype(str).str.strip().fillna("-").str.lower()

# Handle Lang separately - normalize to lowercase for consistency
if "Lang" in df:
    df["Lang"] = df["Lang"].astype(str).str.strip().fillna("-").str.lower()

# ═════════════ 2) Skin mapping ═══════════════
SKINS = {
    "ATD": {
        "de": "autodoc.de",
        "fr": "auto-doc.fr",
        "es": "autodoc.es",
        "gb": "autodoc.co.uk",
        "it": "auto-doc.it"
    },
    "Club": {
        "de": "club.autodoc.de",
        "fr": "club.auto-doc.fr",
        "es": "club.autodoc.es",
        "gb": "club.autodoc.co.uk",
        "it": "club.auto-doc.it"
    },
    "PKW": {
        "de": "pkwteile.de",
        "fr": "piecesauto24.com",
        "es": "recambioscoches.es",
        "gb": "buycarparts.co.uk",
        "it": "autoparti.it"
    },
    "DIREKT": {
        "de": "autoteiledirekt.de",
        "fr": "piecesauto.fr",
        "es": "recambioscoche.es",
        "gb": "onlinecarparts.co.uk",
        "it": "tuttoautoricambi.it"
    },
    "ERSATZ": {
        "de": "autoersatzteile.de",
        "fr": "piecesdiscount24.fr",
        "es": "repuestoscoches24.es",
        "gb": "sparepartstore24.co.uk",
        "it": "pezzidiricambio24.it"
    },
    "PRF": {
        "de": "autoteileprofi.de",
        "fr": "piecesauto-pro.fr",
        "es": "expertoautorecambios.es",
        "gb": "autopartspro.co.uk",
        "it": "	espertoautoricambi.it"
    },
    "EU": {
        "de": "euautoteile.de",
        "fr": "euautopieces.fr",
        "es": "euautorecambios.es",
        "gb": "euspares.co.uk",
        "it": "euautopezzi.it"
    },
    "XXL": {
        "de": "autoteilexxl.de",
        "fr": "24piecesauto.fr",
        "es": "repuestosauto.es",
        "gb": "bestpartstore.co.uk",
        "it": "tuttiautopezzi.it"
    },
    "BVS": {
        "de": "motordoctor.de",
        "fr": "motordoctor.fr",
        "es": "motordoctor.es",
        "gb": "motor-doctor.co.uk",
        "it": "motordoctor.it"
    },
    "TKF": {
        "de": "autotex.de",
        "fr": "autotex.fr",
        "es": "autotex.es",
        "it": "venditapezziauto.it"
    },
    "TOP": {
        "de": "topersatzteile.de",
        "fr": "toppiecesvoiture.fr",
        "es": "toppiezascoches.es",
        "it": "topautoricambi.it"
    },
    "REXBO": {
        "de": "rexbo.de",
        "fr": "rexbo.fr",
        "es": "rexbo.es",
        "gb": "rexbo.co.uk",
        "it": "rexbo.it"
    },
    "ATM": {
        "de": "autoteile-meile.de",
        "fr": "123piecesderechange.fr",
        "es": "recambios-expres.es",
        "gb": "123spareparts.co.uk",
        "it": "shop-ricambiauto.it"
    }
}

# Create set of all own domains for prioritization
OWN_DOMAINS = {dom for skin in SKINS.values() for dom in skin.values()}

def prioritize_domains(domains_list):
    """Sort domains with own domains first, then competitors"""
    own = [d for d in domains_list if d in OWN_DOMAINS]
    competitors = [d for d in domains_list if d not in OWN_DOMAINS]
    return sorted(own) + sorted(competitors)

# ═════════════ 3) Occurrence table ═══════════
def split_urls(txt): return [u.strip() for u in str(txt).split(";") if u.strip() and u.lower()!="nan"]
def host(url):
    if not url.startswith(("http://","https://")): url="http://"+url
    net=urlparse(url).netloc.lower()
    return net[4:] if net.startswith("www.") else net

url_cols=[c for c in df.columns if c.startswith("URL")]
has_main="Main Sources" in df.columns
rows=[]
for _,r in df.iterrows():
    base=dict(Date=r["Date"], Lang=r.get("Lang","-"), Search_Term=r["Search Term"],
              intent=r.get("intent","-"), category=r.get("category","-"), route=r.get("route","-"))
    main=split_urls(r["Main Sources"]) if (has_main and not pd.isna(r["Main Sources"])) else []
    if main:
        for pos,u in enumerate(main,1): rows.append({**base,"Domain":host(u),"URL":u,"Position":pos})
    else:
        for pos,col in enumerate(url_cols,1):
            for u in split_urls(r[col]):
                rows.append({**base,"Domain":host(u),"URL":u,"Position":pos})

occ=pd.DataFrame(rows)
occ["Weight"]=1/occ["Position"]

# ═════════════ 4) Sidebar filters ═══════════
st.sidebar.header("Filters")
def msel(label,key,opts):
    prev=[v for v in st.session_state.get(key,[]) if v in opts]
    return st.sidebar.multiselect(label,sorted(opts),default=prev,key=key)

countries = df["Lang"].dropna().unique() if "Lang" in df else []
f_country = msel("Country","f_country",countries)

dmin,dmax=df["Date"].min(), df["Date"].max()
f_date=st.sidebar.slider("Date range", dmin.to_pydatetime(), dmax.to_pydatetime(),
                         (dmin.to_pydatetime(), dmax.to_pydatetime()),
                         format="YYYY-MM-DD", key="f_date")

tmp=df.copy()
if f_country: tmp=tmp[tmp["Lang"].isin(f_country)]
tmp=tmp[(tmp["Date"]>=f_date[0])&(tmp["Date"]<=f_date[1])]

f_cat   = msel("Category","f_cat",   tmp["category"].dropna().unique() if "category" in tmp else [])
tmp_cat = tmp[tmp["category"].isin(f_cat)] if f_cat else tmp
f_int   = msel("Intent","f_int",     tmp_cat["intent"].dropna().unique() if "intent" in tmp_cat else [])
tmp_int = tmp_cat[tmp_cat["intent"].isin(f_int)] if f_int else tmp_cat
f_route = msel("Route","f_route",    tmp_int["route"].dropna().unique() if "route" in tmp_int else [])

f_skins = st.sidebar.multiselect("Skins", list(SKINS), [])
skin_domains = {dom for s in f_skins for dom in SKINS[s].values()}

# ═════════════ 5) Apply filters (before term) ═
def apply(d):
    if f_country: d=d[d["Lang"].isin(f_country)]
    d=d[(d["Date"]>=f_date[0])&(d["Date"]<=f_date[1])]
    if f_cat:   d=d[d["category"].isin(f_cat)]
    if f_int:   d=d[d["intent"].isin(f_int)]
    if f_route: d=d[d["route"].isin(f_route)]
    return d

df_filt, occ_filt = apply(df), apply(occ)

domain_opts = prioritize_domains(occ_filt["Domain"].unique())
dom_sel = msel("Additional domains (optional)","dom_sel",[d for d in domain_opts if d not in skin_domains])
active_domains = skin_domains | set(dom_sel) if (skin_domains or dom_sel) else set(domain_opts)
occ_filt = occ_filt[occ_filt["Domain"].isin(active_domains)]

all_terms = occ_filt["Search_Term"].unique()
sel_terms= msel("Search Term(s) (controls Coverage & Timeline)","sel_terms", all_terms)

# FIXED: Use consistent filtering basis for all calculations
df_view  = df_filt[df_filt["Search Term"].isin(sel_terms)] if sel_terms else df_filt
occ_view = occ_filt[occ_filt["Search_Term"].isin(sel_terms)] if sel_terms else occ_filt

cov_terms     = sel_terms if sel_terms else all_terms
timeline_term = sel_terms[0] if len(sel_terms)==1 else None

# quick overview of current unique terms
with st.expander(f"Current unique search terms ({df_view['Search Term'].nunique()})",expanded=False):
    st.write(",  ".join(sorted(df_view["Search Term"].unique())) or "*none*")

# ═════════════ 6) Tabs ═══════════════════════
tab_desc, tab_perf, tab_domain, tab_market, tab_ngram, tab_topic = st.tabs(
    ["📊 Description", "🎯 Search Performance", "🌐 Domain Analysis",
     "📈 Market Analysis", "🔍 N-gram Insights", "🎪 Topic Clusters"]
)

# ═════════════ TAB 0 – Description (EN) ══════
with tab_desc:
    st.header("📊 Dashboard Overview")

    # Add visual separator
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        This dashboard analyzes **domain visibility and performance** in AI search results,
        helping you understand how your domains compete in the AI search landscape.

        ### 🎯 Key Metrics Explained
        """)

        metrics_df = pd.DataFrame({
            "Metric": ["Visibility Score", "Share of Voice %", "Avg Position", "Coverage %", "Brand Mentions", "Sources in Response"],
            "Description": [
                "Weighted visibility (1/position sum)",
                "Domain's share of total visibility",
                "Average best position across runs",
                "% of runs where domain appears",
                "Brand mentions in answer text",
                "Direct URL citations count"
            ],
            "Formula": [
                "Σ(1 ÷ position)",
                "(Domain Visibility ÷ Total) × 100",
                "Σ(best positions) ÷ runs",
                "(runs with domain ÷ total) × 100",
                "Regex matches after cleaning",
                "Count of cited URLs"
            ]
        })

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🏢 Your Domains")
        own_domains_df = pd.DataFrame([
            {"Skin": skin, "Domains": ", ".join(domains.values())}
            for skin, domains in SKINS.items()
        ])
        st.dataframe(own_domains_df, use_container_width=True, hide_index=True)

        st.markdown("### 📈 Navigation Guide")
        st.markdown("""
        - **🎯 Search Performance**: KPIs & timelines
        - **🌐 Domain Analysis**: Category breakdowns
        - **⚔️ Domain Comparison**: Head-to-head analysis
        - **🔍 N-gram Insights**: Content analysis
        - **🎪 Topic Clusters**: Theme grouping
        """)

# ═════════════ TAB – Search-Term Performance (Enhanced & Fixed) ════════════
with tab_perf:
    st.header("🎯 Search-Term Performance")

    if occ_view.empty:
        st.warning("⚠️ No data matches current filters.")
    else:
        # Enhanced KPI table with prioritized domains
        st.subheader("📈 Performance Metrics by Domain")

        # Create aggregated data per search term and domain
        grp = occ_view.groupby(["Search_Term","Domain"],as_index=False).agg(
            Avg_Pos=("Position","mean"),
            Visibility=("Weight","sum"),
            Total_Positions=("Position","count")
        ).round(3)

        # Add coverage and source information
        coverage_data = []
        for _,row in grp.iterrows():
            term, dom = row["Search_Term"], row["Domain"]
            sub = df_view[df_view["Search Term"] == term]

            # Count sources in response
            url_list = []
            if has_main:
                for cell in sub["Main Sources"].dropna().astype(str):
                    url_list += split_urls(cell)
            matched = [u for u in url_list if dom in host(u)]
            sources_count = len(matched)
            unique_urls = len(set(matched))

            # Calculate coverage
            term_runs = sub.shape[0]
            present = 0
            if has_main:
                present = sub["Main Sources"].dropna().astype(str).apply(
                    lambda s: any(dom in host(u) for u in split_urls(s))
                ).sum()

            coverage_data.append({
                "Search_Term": term,
                "Domain": dom,
                "Sources_Response": sources_count,
                "Unique_URLs": unique_urls,
                "Coverage": f"{present}/{term_runs}"
            })

        coverage_df = pd.DataFrame(coverage_data)
        grp = grp.merge(coverage_df, on=["Search_Term", "Domain"], how="left")

        # Create a more readable table format
        if not grp.empty:
            # FIXED: Calculate total for search terms correctly
            if len(sel_terms) == 1:
                total_search_terms_for_summary = 1
                total_runs_for_summary = df_view.shape[0]
            else:
                total_search_terms_for_summary = df_filt["Search Term"].nunique()
                total_runs_for_summary = df_filt.shape[0]

            # Get all domains and prioritize own domains
            all_domains = sorted(grp["Domain"].unique())
            prioritized_domains = prioritize_domains(all_domains)

            # Create summary table per domain across all search terms
            summary_data = []

            for domain in prioritized_domains:
                domain_data = grp[grp["Domain"] == domain]
                if not domain_data.empty:
                    # Determine domain type
                    domain_type = "Skin" if domain in OWN_DOMAINS else "Competitor"

                    if len(sel_terms) == 1:
                        # For single search term: domain appears in the selected term
                        search_terms_covered = 1
                        coverage_percentage = 100.0

                        # Calculate run coverage - how many actual runs this domain appears in
                        domain_occ_data = occ_view[occ_view["Domain"] == domain]
                        runs_covered = domain_occ_data["Date"].nunique() if not domain_occ_data.empty else 0
                        run_coverage_pct = (runs_covered / total_runs_for_summary) * 100 if total_runs_for_summary > 0 else 0
                        run_coverage_display = f"{runs_covered}/{total_runs_for_summary}"
                    else:
                        # For multiple terms: use unique search terms logic
                        search_terms_covered = len(domain_data)
                        coverage_percentage = (search_terms_covered / total_search_terms_for_summary) * 100

                        # For multiple terms, run coverage is based on unique date+term combinations
                        domain_occ_data = apply(occ)[apply(occ)["Domain"] == domain]
                        if len(sel_terms) > 1:
                            domain_occ_data = occ_view[occ_view["Domain"] == domain]
                        runs_covered = domain_occ_data[["Date","Search_Term"]].drop_duplicates().shape[0] if not domain_occ_data.empty else 0
                        run_coverage_pct = (runs_covered / total_runs_for_summary) * 100 if total_runs_for_summary > 0 else 0
                        run_coverage_display = f"{runs_covered}/{total_runs_for_summary}"

                    summary_data.append({
                        "Domain": domain,
                        "Type": domain_type,
                        "Avg_Position": round(domain_data["Avg_Pos"].mean(), 2),
                        "Total_Visibility": round(domain_data["Visibility"].sum(), 3),
                        "Total_Sources": domain_data["Sources_Response"].sum(),
                        "Unique_URLs": domain_data["Unique_URLs"].sum(),
                        "Search_Terms_Covered": f"{search_terms_covered}/{total_search_terms_for_summary}",
                        "Coverage_%": round(coverage_percentage, 1),
                        "Run_Coverage": run_coverage_display,
                        "Run_Coverage_%": round(run_coverage_pct, 1),
                        "Coverage_Count": search_terms_covered  # For sorting
                    })

            summary_df = pd.DataFrame(summary_data)

            # Display summary table
            st.markdown("#### 🎯 Domain Performance Summary")
            if not summary_df.empty:
                # Separate own domains and competitors
                own_domains_df = summary_df[summary_df["Type"] == "Skin"].copy()
                competitor_domains_df = summary_df[summary_df["Type"] == "Competitor"].copy()

                # Sort own domains by coverage (descending)
                own_domains_df = own_domains_df.sort_values("Coverage_Count", ascending=False)

                # Sort competitors by coverage (descending)
                competitor_domains_df = competitor_domains_df.sort_values("Coverage_Count", ascending=False)

                # Combine: Own domains first, then competitors
                final_df = pd.concat([own_domains_df, competitor_domains_df], ignore_index=True)

                # Remove the helper column before display
                display_df = final_df.drop("Coverage_Count", axis=1)

                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Display detailed breakdown
            st.markdown("#### 📊 Detailed Performance by Search Term")

            # Create a pivot-like view but more readable using consistent data
            detailed_table = grp.pivot_table(
                index="Search_Term",
                columns="Domain",
                values=["Avg_Pos", "Visibility", "Sources_Response"],
                fill_value=0
            ).round(3)

            # Reorder columns to show own domains first
            if not detailed_table.empty:
                new_columns = []
                for metric in ["Avg_Pos", "Visibility", "Sources_Response"]:
                    for domain in prioritized_domains:
                        if (metric, domain) in detailed_table.columns:
                            new_columns.append((metric, domain))

                if new_columns:
                    detailed_table = detailed_table[new_columns]

                st.dataframe(detailed_table, use_container_width=True)
            else:
                st.info("No detailed data available for current selection.")
        else:
            st.warning("⚠️ No performance data available for current filters.")

    # Enhanced Domain Coverage Chart - FIXED to use consistent data
    st.subheader("📊 Domain Coverage Analysis")

    # FIXED: Use consistent basis depending on search term selection
    if len(sel_terms) == 1:
        # Single search term - show coverage for just that term
        selected_term = sel_terms[0]
        analysis_base_df = df_filt[df_filt["Search Term"] == selected_term]
        analysis_base_occ = apply(occ)[apply(occ)["Search_Term"] == selected_term]
        total_unique_queries = 1  # Only one search term
        total_runs = analysis_base_df.shape[0]
        st.info(f"📊 **Coverage Analysis for:** '{selected_term}' ({total_runs} runs)")
    else:
        # Multiple or all terms
        analysis_base_df = df_filt
        analysis_base_occ = apply(occ)
        total_unique_queries = df_filt["Search Term"].nunique()
        total_runs = df_filt.shape[0]
        if len(sel_terms) > 1:
            analysis_base_df = df_view
            analysis_base_occ = occ_view
            total_unique_queries = df_view["Search Term"].nunique()
            total_runs = df_view.shape[0]
            st.info(f"📊 **Coverage Analysis for {len(sel_terms)} selected search terms:** {total_runs} total runs")
        else:
            st.info(f"📊 **Coverage Analysis:** {total_unique_queries} unique search terms, {total_runs} total runs")

    # Bestimme welche Domains angezeigt werden sollen (works for all cases)
    if skin_domains or dom_sel:
        # FILTER skin domains by selected country/language
        filtered_skin_domains = skin_domains
        if skin_domains and f_country:
            # Map country codes to skin language keys
            country_to_lang = {"de": "de", "fr": "fr", "es": "es", "gb": "gb", "it": "it"}

            filtered_skin_domains = set()
            for skin_name in f_skins:
                if skin_name in SKINS:
                    skin_domains_dict = SKINS[skin_name]
                    for country in f_country:
                        lang_key = country_to_lang.get(country.lower())
                        if lang_key and lang_key in skin_domains_dict:
                            filtered_skin_domains.add(skin_domains_dict[lang_key])

        # Wenn Domains gefiltert sind, zeige nur diese
        domains_to_analyze = filtered_skin_domains | set(dom_sel)
    else:
        # Wenn keine Domain-Filter, zeige alle Domains die vorkommen
        domains_to_analyze = set(analysis_base_occ["Domain"].unique())

    # Enhanced coverage calculations using consistent data
    coverage_data = []
    for domain in domains_to_analyze:
        domain_data = analysis_base_occ[analysis_base_occ["Domain"] == domain]

        if len(sel_terms) == 1:
            # For single search term: count actual runs where domain appears
            unique_queries_covered = 1 if not domain_data.empty else 0
            unique_coverage_pct = 100.0 if not domain_data.empty else 0.0

            # Count unique dates where this domain appears for this search term
            runs_covered = domain_data["Date"].nunique() if not domain_data.empty else 0
            runs_coverage_pct = (runs_covered / total_runs) * 100 if total_runs > 0 else 0
        else:
            # For multiple terms: use original logic
            unique_queries_covered = domain_data["Search_Term"].nunique()
            unique_coverage_pct = (unique_queries_covered / total_unique_queries) * 100

            # All runs coverage using consistent data
            runs_covered = domain_data[["Date","Search_Term"]].drop_duplicates().shape[0]
            runs_coverage_pct = (runs_covered / total_runs) * 100

        coverage_data.append({
            "Domain": domain,
            "Unique_Queries_Covered": f"{unique_queries_covered}/{total_unique_queries}",
            "Unique_Coverage_%": unique_coverage_pct,
            "All_Runs_Covered": f"{runs_covered}/{total_runs}",
            "All_Runs_Coverage_%": runs_coverage_pct,
            "Is_Own_Domain": domain in OWN_DOMAINS
        })

    # Zeige auch Domains mit 0% Coverage wenn sie in filtered skin_domains sind
    if skin_domains and f_country:
        # Use the same filtering logic as above
        country_to_lang = {"de": "de", "fr": "fr", "es": "es", "gb": "gb", "it": "it"}
        filtered_skin_domains_for_zero = set()
        for skin_name in f_skins:
            if skin_name in SKINS:
                skin_domains_dict = SKINS[skin_name]
                for country in f_country:
                    lang_key = country_to_lang.get(country.lower())
                    if lang_key and lang_key in skin_domains_dict:
                        filtered_skin_domains_for_zero.add(skin_domains_dict[lang_key])

        for domain in filtered_skin_domains_for_zero:
            if domain not in [item["Domain"] for item in coverage_data]:
                coverage_data.append({
                    "Domain": domain,
                    "Unique_Queries_Covered": f"0/{total_unique_queries}",
                    "Unique_Coverage_%": 0.0,
                    "All_Runs_Covered": f"0/{total_runs}",
                    "All_Runs_Coverage_%": 0.0,
                    "Is_Own_Domain": True
                })
    elif skin_domains:
        # Original logic for domains with 0% coverage when no country filter
        for domain in skin_domains:
            if domain not in [item["Domain"] for item in coverage_data]:
                coverage_data.append({
                    "Domain": domain,
                    "Unique_Queries_Covered": f"0/{total_unique_queries}",
                    "Unique_Coverage_%": 0.0,
                    "All_Runs_Covered": f"0/{total_runs}",
                    "All_Runs_Coverage_%": 0.0,
                    "Is_Own_Domain": True
                })

    if coverage_data:
        cov_df = pd.DataFrame(coverage_data)
        # Sort by own domains first, then by coverage
        cov_df = cov_df.sort_values(["Is_Own_Domain", "All_Runs_Coverage_%"], ascending=[False, False])

        # Display with better formatting
        display_df = cov_df[["Domain", "Unique_Queries_Covered", "Unique_Coverage_%",
                            "All_Runs_Covered", "All_Runs_Coverage_%"]].copy()
        display_df["Unique_Coverage_%"] = display_df["Unique_Coverage_%"].round(1)
        display_df["All_Runs_Coverage_%"] = display_df["All_Runs_Coverage_%"].round(1)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Enhanced visualization
        chart_data = cov_df.head(15)
        if not chart_data.empty:
            chart = alt.Chart(chart_data).mark_bar().encode(
                y=alt.Y("Domain:N", sort=alt.EncodingSortField(field="All_Runs_Coverage_%", order="descending")),
                x=alt.X("All_Runs_Coverage_%:Q", title="Coverage % (All Runs)"),
                color=alt.Color("Is_Own_Domain:N",
                              scale=alt.Scale(domain=[True, False], range=["#2E86AB", "#A23B72"]),
                              legend=alt.Legend(title="Domain Type"))
            ).properties(height=400)

            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("ℹ️ No domains to analyze. Please select domains in the sidebar.")

    # FIXED: Missing Coverage Analysis for Own Domains
    st.subheader("🔍 Missing Coverage Analysis")

    # FIXED: Get all search terms from consistent analysis base
    if len(sel_terms) == 1:
        all_search_terms = set([sel_terms[0]])  # Only the selected search term
    else:
        all_search_terms = set(analysis_base_df["Search Term"].unique())

    # FIXED: Use EXACTLY the same logic as Coverage Matrix - country-specific domains only
    relevant_own_domains = set()
    if f_country:
        # Use exactly the same logic as matrix analysis
        country_to_lang = {"de": "de", "fr": "fr", "es": "es", "gb": "gb", "it": "it"}
        skins_to_use = f_skins if f_skins else list(SKINS.keys())

        for skin_name in skins_to_use:
            if skin_name in SKINS:
                skin_domains_dict = SKINS[skin_name]
                for country in f_country:
                    lang_key = country_to_lang.get(country.lower())
                    if lang_key and lang_key in skin_domains_dict:
                        relevant_own_domains.add(skin_domains_dict[lang_key])
    elif skin_domains:
        # If no country filter but skin domains selected, use skin domains
        relevant_own_domains = skin_domains
    else:
        # Fallback to all own domains
        relevant_own_domains = OWN_DOMAINS

    if relevant_own_domains:
        st.markdown("#### 🚨 Search Terms with Missing Own Domain Coverage")

        # Show which domains are being analyzed for this country - SAME as Coverage Matrix
        if f_country and relevant_own_domains:
            country_names = {"de": "Germany", "fr": "France", "es": "Spain", "gb": "UK", "it": "Italy"}
            selected_countries = [country_names.get(c, c.upper()) for c in f_country]

            if f_skins:
                selected_skins = ", ".join(f_skins)
                st.success(f"📍 **Analyzing {selected_skins} domains for {', '.join(selected_countries)}**: {', '.join(sorted(relevant_own_domains))}")
            else:
                st.success(f"📍 **Analyzing ALL skin domains for {', '.join(selected_countries)}**: {', '.join(sorted(relevant_own_domains))}")


        # FIXED: Find search terms where none of our own domains appear with 100% consistency
        # Process each search term with its language-specific domains
        missing_coverage_data = []
        full_coverage_data = []
        partial_coverage_data = []

        country_to_lang = {"de": "de", "fr": "fr", "es": "es", "gb": "gb", "it": "it"}

        for search_term in all_search_terms:
            # Get all runs for this search term using consistent data
            term_runs_df = analysis_base_df[analysis_base_df["Search Term"] == search_term]
            total_runs_for_term = term_runs_df.shape[0]

            # Get the language/country for this search term
            term_lang = term_runs_df["Lang"].iloc[0] if not term_runs_df.empty and "Lang" in term_runs_df else None

            # Determine which domains to check for this specific search term
            term_relevant_domains = set()

            if f_country and (skin_domains or f_skins):
                # If country filter is active, use the globally relevant domains
                term_relevant_domains = relevant_own_domains
            elif term_lang:
                # No country filter: determine domains based on search term's language
                lang_key = country_to_lang.get(term_lang.lower())
                if lang_key:
                    skins_to_use = f_skins if f_skins else list(SKINS.keys())
                    for skin_name in skins_to_use:
                        if skin_name in SKINS and lang_key in SKINS[skin_name]:
                            term_relevant_domains.add(SKINS[skin_name][lang_key])
            else:
                # Fallback: use all relevant own domains
                term_relevant_domains = relevant_own_domains

            # Skip if no relevant domains for this term
            if not term_relevant_domains:
                continue

            # Get occurrences for this search term using consistent data
            term_occurrences = analysis_base_occ[analysis_base_occ["Search_Term"] == search_term]
            term_domains = set(term_occurrences["Domain"].unique())

            # Check coverage for each relevant domain for this specific search term
            domain_coverage = {}
            for domain in term_relevant_domains:
                domain_term_occurrences = term_occurrences[term_occurrences["Domain"] == domain]
                if not domain_term_occurrences.empty:
                    # Count unique run instances (Date + Search_Term combinations)
                    runs_with_domain = domain_term_occurrences[["Date", "Search_Term"]].drop_duplicates().shape[0]
                    coverage_percentage = (runs_with_domain / total_runs_for_term) * 100 if total_runs_for_term > 0 else 0
                    domain_coverage[domain] = coverage_percentage
                else:
                    domain_coverage[domain] = 0

            # Categorize based on coverage
            domains_with_100_coverage = [d for d, cov in domain_coverage.items() if cov == 100]
            domains_with_partial_coverage = [d for d, cov in domain_coverage.items() if 0 < cov < 100]
            domains_with_no_coverage = [d for d, cov in domain_coverage.items() if cov == 0]

            # Get competitor domains for context
            competitor_domains = [d for d in term_domains if d not in OWN_DOMAINS]

            if domains_with_100_coverage:
                # Full coverage - at least one domain has 100%
                full_coverage_data.append({
                    "Search_Term": search_term,
                    "Domains_100%": ", ".join(sorted(domains_with_100_coverage)),
                    "Domains_Partial": ", ".join(sorted(domains_with_partial_coverage)) if domains_with_partial_coverage else "None",
                    "Domains_Missing": ", ".join(sorted(domains_with_no_coverage)) if domains_with_no_coverage else "None",
                    "Total_Runs": total_runs_for_term,
                    "Competitor_Domains": ", ".join(sorted(competitor_domains)[:3]) + ("..." if len(competitor_domains) > 3 else "")
                })
            elif domains_with_partial_coverage:
                # Partial coverage - some domains appear but not consistently
                partial_coverage_data.append({
                    "Search_Term": search_term,
                    "Domains_Partial": ", ".join([f"{d} ({domain_coverage[d]:.0f}%)" for d in sorted(domains_with_partial_coverage)]),
                    "Domains_Missing": ", ".join(sorted(domains_with_no_coverage)) if domains_with_no_coverage else "None",
                    "Total_Runs": total_runs_for_term,
                    "Competitor_Domains": ", ".join(sorted(competitor_domains)[:3]) + ("..." if len(competitor_domains) > 3 else "")
                })
            else:
                # No coverage - none of our domains appear
                missing_coverage_data.append({
                    "Search_Term": search_term,
                    "Missing_Domains": ", ".join(sorted(term_relevant_domains)),
                    "Total_Runs": total_runs_for_term,
                    "Competitor_Domains": ", ".join(sorted(competitor_domains)[:3]) + ("..." if len(competitor_domains) > 3 else "")
                })

        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Terms with NO own domains", len(missing_coverage_data))
        with col2:
            st.metric("⚠️ Terms with partial coverage", len(partial_coverage_data))
        with col3:
            st.metric("✅ Terms with full coverage", len(full_coverage_data))

        # Show detailed breakdowns
        if missing_coverage_data:
            st.markdown("**🚨 Terms with NO Own Domain Coverage:**")
            missing_df = pd.DataFrame(missing_coverage_data).sort_values("Search_Term")
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

        if partial_coverage_data:
            st.markdown("**⚠️ Terms with Partial Own Domain Coverage:**")
            partial_df = pd.DataFrame(partial_coverage_data).sort_values("Search_Term")
            st.dataframe(partial_df, use_container_width=True, hide_index=True)

        if full_coverage_data:
            st.markdown("**✅ Terms with Full Own Domain Coverage (100%):**")
            full_df = pd.DataFrame(full_coverage_data).sort_values("Search_Term")
            st.dataframe(full_df, use_container_width=True, hide_index=True)

        # Download options
        if missing_coverage_data or partial_coverage_data:
            combined_data = []

            # Add missing coverage data
            for item in missing_coverage_data:
                combined_data.append({
                    "Search_Term": item["Search_Term"],
                    "Coverage_Type": "No Coverage",
                    "Own_Domains_100%": "",
                    "Own_Domains_Partial": "",
                    "Own_Domains_Missing": item["Missing_Domains"],
                    "Total_Runs": item["Total_Runs"],
                    "Competitor_Domains": item["Competitor_Domains"]
                })

            # Add partial coverage data
            for item in partial_coverage_data:
                combined_data.append({
                    "Search_Term": item["Search_Term"],
                    "Coverage_Type": "Partial Coverage",
                    "Own_Domains_100%": "",
                    "Own_Domains_Partial": item["Domains_Partial"],
                    "Own_Domains_Missing": item["Domains_Missing"],
                    "Total_Runs": item["Total_Runs"],
                    "Competitor_Domains": item["Competitor_Domains"]
                })

            # Add full coverage data
            for item in full_coverage_data:
                combined_data.append({
                    "Search_Term": item["Search_Term"],
                    "Coverage_Type": "Full Coverage",
                    "Own_Domains_100%": item["Domains_100%"],
                    "Own_Domains_Partial": item["Domains_Partial"],
                    "Own_Domains_Missing": item["Domains_Missing"],
                    "Total_Runs": item["Total_Runs"],
                    "Competitor_Domains": item["Competitor_Domains"]
                })

            if combined_data:
                combined_df = pd.DataFrame(combined_data)
                st.download_button(
                    "📥 Download Complete Coverage Analysis",
                    combined_df.to_csv(index=False).encode("utf-8"),
                    "complete_coverage_analysis.csv",
                    "text/csv",
                    key="dl_complete_coverage"
                )

        if not missing_coverage_data and not partial_coverage_data:
            st.success("🎉 All search terms have complete 100% coverage from your selected domains!")

    else:
        st.info("ℹ️ Select skin domains in the sidebar to see missing coverage analysis.")

    # FIXED: Coverage Matrix with Percentage for Country-Specific Domains
    if f_country and len(analysis_base_df) > 0:
        st.subheader("📋 Coverage Matrix: Country-Specific Domains vs Search Terms")

        # WICHTIG: Zeige alle Skin-Domains für das gewählte Land (oder nur gewählte Skins)
        matrix_domains = set()
        country_to_lang = {"de": "de", "fr": "fr", "es": "es", "gb": "gb", "it": "it"}

        # Bestimme welche Skins verwendet werden sollen
        skins_to_use = f_skins if f_skins else list(SKINS.keys())  # Alle Skins wenn keine ausgewählt

        for skin_name in skins_to_use:
            if skin_name in SKINS:
                skin_domains_dict = SKINS[skin_name]
                for country in f_country:
                    lang_key = country_to_lang.get(country.lower())
                    if lang_key and lang_key in skin_domains_dict:
                        matrix_domains.add(skin_domains_dict[lang_key])

        if not matrix_domains:
            st.warning("⚠️ No domains found for selected country.")
        else:
            # Show which domains are being analyzed
            country_names = {"de": "Germany", "fr": "France", "es": "Spain", "gb": "UK", "it": "Italy"}
            selected_countries = [country_names.get(c, c.upper()) for c in f_country]

            if f_skins:
                selected_skins = ", ".join(f_skins)
                st.success(f"📍 **Analyzing {selected_skins} domains for {', '.join(selected_countries)}**: {', '.join(sorted(matrix_domains))}")
            else:
                st.success(f"📍 **Analyzing ALL skin domains for {', '.join(selected_countries)}**: {', '.join(sorted(matrix_domains))}")

            # Create matrix data for all search terms (showing percentage coverage)
            matrix_data = []

            for term in sorted(all_search_terms):
                term_row = {"Search_Term": term}

                # Get all runs for this search term from consistent base
                term_runs_df = analysis_base_df[analysis_base_df["Search Term"] == term]
                total_runs_for_term = term_runs_df.shape[0]

                # Get occurrences for this search term using consistent data
                term_occurrences = analysis_base_occ[analysis_base_occ["Search_Term"] == term]

                for domain in sorted(matrix_domains):
                    # Count how many runs this domain appeared in for this search term
                    domain_term_occurrences = term_occurrences[term_occurrences["Domain"] == domain]

                    if not domain_term_occurrences.empty and total_runs_for_term > 0:
                        # Count unique run instances (Date + Search_Term combinations)
                        runs_with_domain = domain_term_occurrences[["Date", "Search_Term"]].drop_duplicates().shape[0]
                        coverage_percentage = (runs_with_domain / total_runs_for_term) * 100

                        if coverage_percentage == 100:
                            term_row[domain] = "100%"
                        else:
                            term_row[domain] = f"{coverage_percentage:.0f}% ({runs_with_domain}/{total_runs_for_term})"
                    else:
                        # Domain never appeared for this search term
                        term_row[domain] = "❌"

                matrix_data.append(term_row)

            if matrix_data:
                matrix_df = pd.DataFrame(matrix_data)

                # Calculate coverage statistics
                total_combinations = len(all_search_terms) * len(matrix_domains)
                covered_combinations = 0
                full_coverage_combinations = 0

                for row in matrix_data:
                    for key, value in row.items():
                        if key != "Search_Term":
                            if value != "❌":
                                covered_combinations += 1
                            if value == "100%":
                                full_coverage_combinations += 1

                coverage_percentage = (covered_combinations / total_combinations) * 100 if total_combinations > 0 else 0
                full_coverage_percentage = (full_coverage_combinations / total_combinations) * 100 if total_combinations > 0 else 0

                st.info(f"📊 **Coverage Matrix:** {covered_combinations}/{total_combinations} combinations have coverage ({coverage_percentage:.1f}%) | {full_coverage_combinations} combinations have 100% coverage ({full_coverage_percentage:.1f}%)")

                # Display matrix (all domains since they are country-specific now)
                st.dataframe(matrix_df, use_container_width=True, hide_index=True)

                # Download full matrix
                st.download_button(
                    "📥 Download Full Coverage Matrix with Percentages",
                    matrix_df.to_csv(index=False).encode("utf-8"),
                    "coverage_matrix_percentages.csv",
                    "text/csv",
                    key="dl_coverage_matrix_main"
                )
            else:
                st.info("ℹ️ No matrix data available for current selection.")
    elif len(analysis_base_df) > 0:
        st.subheader("📋 Coverage Matrix: Country-Specific Domains vs Search Terms")
        st.info("ℹ️ Please select a **Country** in the sidebar to display the coverage matrix.")
    else:
        st.info("ℹ️ Select skin domains in the sidebar to see missing coverage analysis.")

    # Timeline section (FIXED: Chart with compact legend on the right)
    st.subheader("📈 Position Timeline")
    if timeline_term is None:
        st.info("ℹ️ Select exactly **one** Search Term above to display its timeline.")
    else:
        pos=occ_view[occ_view["Search_Term"]==timeline_term]
        if not pos.empty:
            pts_all=pos[["Domain","Date","Position"]].drop_duplicates()
            best   =pos.groupby(["Domain","Date"],as_index=False).agg(Position=("Position","min"))

            # Prioritize own domains in legend
            doms = prioritize_domains(best["Domain"].unique())
            cmap=color_map(doms)

            # Calculate average positions for legend table
            domain_stats = best.groupby("Domain").agg(
                Avg_Position=("Position", "mean"),
                Appearances=("Position", "count")
            ).round(2)

            # Add colors and domain type to stats
            domain_stats["Color"] = domain_stats.index.map(lambda x: cmap[x])
            domain_stats["Type"] = domain_stats.index.map(lambda x: "Own" if x in OWN_DOMAINS else "Competitor")
            domain_stats = domain_stats.reindex(doms)  # Sort by priority

            # FIXED: Get unique dates and sort them to ensure proper spacing
            unique_dates = sorted(best["Date"].unique())

            # Create chart without legend
            line=alt.Chart(best).mark_line(point=False).encode(
                x=alt.X("Date:T",
                       scale=alt.Scale(domain=[unique_dates[0], unique_dates[-1]]),
                       axis=alt.Axis(format="%b %d", labelAngle=-45)),
                y=alt.Y("Position:Q",
                       scale=alt.Scale(domain=[1,pts_all["Position"].max()],reverse=True,nice=False),
                       axis=alt.Axis(values=list(range(1,pts_all["Position"].max()+1)))),
                color=alt.Color("Domain:N",scale=alt.Scale(domain=doms,range=[cmap[d] for d in doms]),
                                legend=None),  # No legend on chart
                detail="Domain:N"
            )

            dots=alt.Chart(pts_all).mark_circle(size=120).encode(
                x=alt.X("Date:T",
                       scale=alt.Scale(domain=[unique_dates[0], unique_dates[-1]])),
                y="Position:Q",
                color=alt.Color("Domain:N",scale=alt.Scale(domain=doms,range=[cmap[d] for d in doms]),legend=None),
                tooltip=["Domain", alt.Tooltip("Date:T", format="%Y-%m-%d"), "Position"]
            )

            combined_chart = (line + dots).properties(
                height=400,
                width="container"
            ).resolve_scale(
                color='independent'
            )

            # FIXED: Dynamic layout - more space for chart, compact legend
            num_domains = len(doms)
            if num_domains <= 8:
                col_ratio = [0.8, 0.2]    # More space for chart when few domains
            else:
                col_ratio = [0.75, 0.25]  # Slightly more space for legend with many domains

            # Layout: Chart on left, Compact legend on right
            col1, col2 = st.columns(col_ratio)

            with col1:
                st.altair_chart(combined_chart, use_container_width=True)

            with col2:
                # FIXED: Create compact legend with actual chart colors using st.markdown
                st.markdown("**🎯 Legend**")

                # Create compact legend with actual colors from chart
                for domain in doms:
                    if domain in domain_stats.index:
                        stats = domain_stats.loc[domain]
                        color = stats["Color"]
                        domain_type = "Own" if domain in OWN_DOMAINS else "Competitor"

                        # Create a simple text-based legend entry
                        st.markdown(f"""
                        <div style='margin-bottom: 6px; font-size: 12px;'>
                            <span style='color: {color}; font-size: 16px;'>●</span>
                            <strong>{domain}</strong>
                            <span style='color: #888; font-size: 11px;'>({stats["Avg_Position"]:.1f})</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Add compact explanation
                st.markdown("""
                <div style='font-size: 10px; color: #666; margin-top: 15px; font-style: italic;'>
                Each domain has a unique color.<br>
                Numbers show average position.
                </div>
                """, unsafe_allow_html=True)

            # URL × Date position matrix with prioritized domains
            st.markdown("### 🔗 URL Position Matrix")
            url_best = pos.groupby(["URL","Date"],as_index=False).agg(Position=("Position","min"))
            url_best["DomainCol"]=url_best["URL"].apply(host)
            piv=url_best.pivot(index="URL",columns="Date",values="Position")
            piv=piv.sort_index(axis=1,ascending=False)
            piv.columns=[d.strftime("%Y-%m-%d") for d in piv.columns]

            # Add domain column and sort by own domains first
            piv.insert(0, "Domain", [host(u) for u in piv.index])
            piv["Is_Own"] = piv["Domain"].apply(lambda x: x in OWN_DOMAINS)
            piv = piv.sort_values(["Is_Own", "Domain"], ascending=[False, True])
            piv = piv.drop("Is_Own", axis=1)

            st.dataframe(piv, use_container_width=True)

            st.download_button("📥 Download URL matrix",
                               piv.reset_index().to_csv(index=False).encode("utf-8"),
                               f"{timeline_term}_url_positions.csv","text/csv",
                               key="dl_url_matrix")

    # NEW: URL Performance Analysis - MOVED HERE after timeline, before data export
    st.subheader("🔗 URL Performance Analysis")

    # Check if any filters are active
    filters_active = bool(f_country or f_cat or f_int or f_route or f_skins or dom_sel or sel_terms)

    if not filters_active:
        st.info("ℹ️ Please apply at least one filter (Country, Category, Intent, Route, Skins, or Domains) to see URL performance analysis.")
    elif occ_view.empty:
        st.warning("⚠️ No URL data matches current filters.")
    else:
        # Analyze URL performance using filtered data
        url_performance = occ_view.groupby("URL").agg(
            Appearances=("Position", "count"),
            Avg_Position=("Position", "mean"),
            Best_Position=("Position", "min"),
            Total_Visibility=("Weight", "sum"),
            Unique_Search_Terms=("Search_Term", "nunique"),
            Unique_Dates=("Date", "nunique")
        ).round(3)

        # Add domain and type information
        url_performance["Domain"] = url_performance.index.map(host)
        url_performance["Type"] = url_performance["Domain"].map(lambda x: "Skin" if x in OWN_DOMAINS else "Competitor")

        # Calculate coverage percentage
        total_possible_appearances = len(occ_view["Search_Term"].unique()) * len(occ_view["Date"].unique())
        url_performance["Coverage_%"] = (url_performance["Appearances"] / total_possible_appearances * 100).round(1)

        # Reorder columns for better display
        url_performance = url_performance[["Domain", "Type", "Appearances", "Avg_Position", "Best_Position",
                                         "Total_Visibility", "Coverage_%", "Unique_Search_Terms", "Unique_Dates"]]

        # Sort by own domains first, then by appearances (frequency)
        url_performance = url_performance.sort_values(["Type", "Appearances"], ascending=[True, False])

        # Show filter summary
        active_filters = []
        if f_country:
            country_names = {"de": "Germany", "fr": "France", "es": "Spain", "gb": "UK", "it": "Italy"}
            countries_display = [country_names.get(c, c.upper()) for c in f_country]
            active_filters.append(f"Country: {', '.join(countries_display)}")
        if f_cat:
            active_filters.append(f"Category: {', '.join(f_cat)}")
        if f_int:
            active_filters.append(f"Intent: {', '.join(f_int)}")
        if f_route:
            active_filters.append(f"Route: {', '.join(f_route)}")
        if f_skins:
            active_filters.append(f"Skins: {', '.join(f_skins)}")
        if dom_sel:
            active_filters.append(f"Additional Domains: {', '.join(dom_sel[:3])}{'...' if len(dom_sel) > 3 else ''}")
        if sel_terms:
            active_filters.append(f"Search Terms: {', '.join(sel_terms[:2])}{'...' if len(sel_terms) > 2 else ''}")

        st.info(f"📊 **Active Filters**: {' | '.join(active_filters)}")

        # Split for metrics but combine for display
        own_urls = url_performance[url_performance["Type"] == "Skin"]
        competitor_urls = url_performance[url_performance["Type"] == "Competitor"]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏢 Own URLs", len(own_urls))
        with col2:
            st.metric("⚔️ Competitor URLs", len(competitor_urls))
        with col3:
            if not own_urls.empty:
                st.metric("📊 Own URLs Avg Appearances", f"{own_urls['Appearances'].mean():.1f}")
            else:
                st.metric("📊 Own URLs Avg Appearances", "0")
        with col4:
            if not competitor_urls.empty:
                st.metric("📊 Competitor URLs Avg Appearances", f"{competitor_urls['Appearances'].mean():.1f}")
            else:
                st.metric("📊 Competitor URLs Avg Appearances", "0")

        # FIXED: Single combined table showing all URLs (Skins first, then Competitors)
        st.markdown("#### 🔗 URL Performance Rankings")
        st.info(f"📈 **Showing {len(url_performance)} unique URLs** from {url_performance['Appearances'].sum()} total appearances (Skins first, then sorted by frequency)")

        # Display the complete table (already sorted: Skins first, then by appearances)
        st.dataframe(url_performance, use_container_width=True, hide_index=False)

        # Visualization: Top URLs by appearances - FIXED: Show domains instead of URLs
        st.markdown("#### 📊 URL Performance Visualization")

        # Show top URLs from the combined table
        chart_data = url_performance.head(15)  # Top 15 URLs overall

        if not chart_data.empty:
            # Use domains for display, keep URLs in tooltip
            chart_data_display = chart_data.reset_index()

            chart = alt.Chart(chart_data_display).mark_bar().encode(
                y=alt.Y("Domain:N", sort=alt.EncodingSortField(field="Appearances", order="descending"),
                       title="Domain"),
                x=alt.X("Appearances:Q", title="Number of Appearances"),
                color=alt.Color("Type:N",
                              scale=alt.Scale(domain=["Skin", "Competitor"], range=["#2E86AB", "#A23B72"]),
                              legend=alt.Legend(title="URL Type")),
                tooltip=["URL:N", "Domain:N", "Type:N", "Appearances:Q", "Avg_Position:Q", "Total_Visibility:Q"]
            ).properties(height=400)

            st.altair_chart(chart, use_container_width=True)

        # Single download option
        st.download_button(
            "📥 Download Complete URL Performance Analysis",
            url_performance.reset_index().to_csv(index=False).encode("utf-8"),
            "complete_url_performance.csv",
            "text/csv",
            key="dl_url_performance"
        )

    # Raw data export
    st.markdown("---")
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_view.head(), use_container_width=True)
    with col2:
        st.download_button("📥 Download Filtered Data",
                          df_view.to_csv(index=False).encode("utf-8"),
                          "filtered_raw_data.csv","text/csv", key="dl_perf_raw")

# ═════════════ TAB – Domain Analysis ═════════════
with tab_domain:
    st.header("🌐 Domain Analysis")

    if occ_view.empty:
        st.warning("⚠️ No data matches current filters.")
    else:
        # FIXED: Use consistent filtered basis (respects search term selection)
        base_data = df_view  # Use filtered data that respects search term selection
        base_occurrences = occ_view  # Use filtered occurrences that respect search term selection

        st.subheader("📊 Domain Performance Overview")

        # Show filter summary like in other tabs
        active_filters = []
        if f_country:
            country_names = {"de": "Germany", "fr": "France", "es": "Spain", "gb": "UK", "it": "Italy"}
            countries_display = [country_names.get(c, c.upper()) for c in f_country]
            active_filters.append(f"Country: {', '.join(countries_display)}")
        if f_cat:
            active_filters.append(f"Category: {', '.join(f_cat)}")
        if f_int:
            active_filters.append(f"Intent: {', '.join(f_int)}")
        if f_route:
            active_filters.append(f"Route: {', '.join(f_route)}")
        if f_skins:
            active_filters.append(f"Skins: {', '.join(f_skins)}")
        if dom_sel:
            active_filters.append(f"Additional Domains: {', '.join(dom_sel[:3])}{'...' if len(dom_sel) > 3 else ''}")
        if sel_terms:
            active_filters.append(f"Search Terms: {', '.join(sel_terms[:2])}{'...' if len(sel_terms) > 2 else ''}")

        if active_filters:
            st.info(f"📊 **Active Filters**: {' | '.join(active_filters)}")

        # Domain visibility analysis using consistent filtered basis
        domain_stats = base_occurrences.groupby("Domain").agg(
            Total_Visibility=("Weight", "sum"),
            Avg_Position=("Position", "mean"),
            Total_Appearances=("Position", "count"),
            Unique_Search_Terms=("Search_Term", "nunique"),
            Unique_Dates=("Date", "nunique")
        ).round(3)

        # Calculate coverage percentages using consistent filtered basis
        total_search_terms = base_data["Search Term"].nunique()
        total_runs = base_data.shape[0]

        domain_stats["Coverage_Unique_Terms_%"] = (domain_stats["Unique_Search_Terms"] / total_search_terms * 100).round(1)
        domain_stats["Domain_Type"] = domain_stats.index.map(lambda x: "Own" if x in OWN_DOMAINS else "Competitor")

        # Sort by own domains first, then by visibility
        domain_stats = domain_stats.sort_values(["Domain_Type", "Total_Visibility"], ascending=[True, False])

        # Add explanations for key metrics
        st.info("ℹ️ **Total_Appearances** = Number of times this domain appeared across all filtered search results (higher = more frequent)")
        st.info("ℹ️ **Total_Visibility** = Quality-weighted score: Σ(1 ÷ position). Position 1 = 1.0 points, Position 5 = 0.2 points, etc. (higher = better positions)")

        st.dataframe(domain_stats, use_container_width=True)

        # FIXED: Visualization that prioritizes own domains
        st.subheader("📊 Domain Performance Visualization")

        # Separate own and competitor domains
        own_domains_stats = domain_stats[domain_stats["Domain_Type"] == "Own"]
        competitor_domains_stats = domain_stats[domain_stats["Domain_Type"] == "Competitor"]

        # Take top own domains and top competitors to ensure own domains are visible
        top_own = own_domains_stats.head(8)  # Top 8 own domains
        top_competitors = competitor_domains_stats.head(7)  # Top 7 competitors

        # Combine for visualization
        chart_data = pd.concat([top_own, top_competitors]) if not top_own.empty else top_competitors.head(15)

        if not chart_data.empty:
            chart = alt.Chart(chart_data.reset_index()).mark_bar().encode(
                y=alt.Y("Domain:N", sort=alt.EncodingSortField(field="Total_Visibility", order="descending")),
                x=alt.X("Total_Visibility:Q", title="Total Visibility Score"),
                color=alt.Color("Domain_Type:N",
                              scale=alt.Scale(domain=["Own", "Competitor"], range=["#2E86AB", "#A23B72"]),
                              legend=alt.Legend(title="Domain Type")),
                tooltip=["Domain:N", "Domain_Type:N", "Total_Visibility:Q", "Avg_Position:Q", "Total_Appearances:Q", "Coverage_Unique_Terms_%:Q"]
            ).properties(height=400)

            st.altair_chart(chart, use_container_width=True)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏢 Own Domains", len(own_domains_stats))
            with col2:
                st.metric("⚔️ Competitor Domains", len(competitor_domains_stats))
            with col3:
                if not own_domains_stats.empty:
                    st.metric("📊 Own Avg Visibility", f"{own_domains_stats['Total_Visibility'].mean():.1f}")
                else:
                    st.metric("📊 Own Avg Visibility", "0")
            with col4:
                if not competitor_domains_stats.empty:
                    st.metric("📊 Competitor Avg Visibility", f"{competitor_domains_stats['Total_Visibility'].mean():.1f}")
                else:
                    st.metric("📊 Competitor Avg Visibility", "0")
        else:
            st.info("ℹ️ No domain data available for visualization.")

        # Download option
        st.download_button(
            "📥 Download Domain Analysis",
            domain_stats.reset_index().to_csv(index=False).encode("utf-8"),
            "domain_analysis.csv",
            "text/csv",
            key="dl_domain_analysis"
        )

        # INTEGRATED: Domain Comparison (formerly separate tab)
        st.markdown("---")
        st.subheader("⚔️ Domain Comparison & Position Trends")

        # Domain selection for comparison
        available_domains = prioritize_domains(base_occurrences["Domain"].unique())
        selected_domains = st.multiselect(
            "Select specific domains to compare (leave empty to show all filtered domains):",
            available_domains,
            default=[],  # Empty by default
            key="domain_comparison_integrated"
        )

        # Use selected domains or all filtered domains
        if selected_domains:
            comparison_data = base_occurrences[base_occurrences["Domain"].isin(selected_domains)]
            st.info(f"🎯 **Comparing {len(selected_domains)} selected domains**: {', '.join(selected_domains[:3])}{'...' if len(selected_domains) > 3 else ''}")
        else:
            comparison_data = base_occurrences
            st.info(f"📊 **Showing all {len(available_domains)} filtered domains** (select specific domains above to focus comparison)")

        if not comparison_data.empty:
            # Head-to-head comparison table
            comparison_stats = comparison_data.groupby("Domain").agg(
                Avg_Position=("Position", "mean"),
                Total_Visibility=("Weight", "sum"),
                Search_Terms_Covered=("Search_Term", "nunique"),
                Total_Appearances=("Position", "count")
            ).round(3)

            # Add coverage percentage and domain type
            total_search_terms = base_data["Search Term"].nunique()
            comparison_stats["Coverage_Unique_Terms_%"] = (comparison_stats["Search_Terms_Covered"] / total_search_terms * 100).round(1)
            comparison_stats["Domain_Type"] = comparison_stats.index.map(lambda x: "Own" if x in OWN_DOMAINS else "Competitor")

            # Sort by own domains first
            comparison_stats = comparison_stats.sort_values(["Domain_Type", "Total_Visibility"], ascending=[True, False])

            st.markdown("#### 📊 Head-to-Head Comparison")
            st.dataframe(comparison_stats, use_container_width=True)

            # Position comparison over time (only if we have time data)
            if len(comparison_data["Date"].unique()) > 1:
                st.markdown("#### 📈 Position Trends Over Time")
                time_comparison = comparison_data.groupby(["Domain", "Date"])["Position"].mean().reset_index()

                # Limit to top domains for cleaner visualization
                if selected_domains:
                    chart_domains = selected_domains
                else:
                    # Show top 10 domains by visibility for cleaner chart
                    top_domains_for_chart = comparison_stats.head(10).index.tolist()
                    time_comparison = time_comparison[time_comparison["Domain"].isin(top_domains_for_chart)]
                    chart_domains = top_domains_for_chart

                if not time_comparison.empty:
                    chart = alt.Chart(time_comparison).mark_line(point=True).encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Position:Q", scale=alt.Scale(reverse=True), title="Average Position"),
                        color=alt.Color("Domain:N",
                                      scale=alt.Scale(domain=prioritize_domains(chart_domains),
                                                    range=_PALETTE[:len(chart_domains)])),
                        tooltip=["Domain:N", alt.Tooltip("Date:T", format="%Y-%m-%d"), "Position:Q"]
                    ).properties(height=400)

                    st.altair_chart(chart, use_container_width=True)

                    if not selected_domains and len(available_domains) > 10:
                        st.info(f"ℹ️ Showing top 10 domains for cleaner visualization. Select specific domains above to compare others.")
                else:
                    st.info("ℹ️ No time trend data available for current selection.")
            else:
                st.info("ℹ️ Position trends require data from multiple dates.")

            # Download comparison data
            st.download_button(
                "📥 Download Domain Comparison Data",
                comparison_stats.reset_index().to_csv(index=False).encode("utf-8"),
                "domain_comparison.csv",
                "text/csv",
                key="dl_domain_comparison"
            )
        else:
            st.warning("⚠️ No comparison data available for current selection.")

# ═════════════ TAB – Market Analysis ═════════════
with tab_market:
    st.header("📈 Precise Brand Analysis - AI Visibility Intelligence")

    if occ_view.empty:
        st.warning("⚠️ No data matches current filters.")
    else:
        # Use consistent filtered data
        base_data = df_view
        base_occurrences = occ_view

        # Brand Classification System - IMPROVED WITH SKINS LOGIC
        def classify_brand(domain):
            """Extract brand name from domain with SKINS-aware classification"""
            if pd.isna(domain) or not domain:
                return 'Unknown'

            domain_clean = domain.lower().strip()

            # PRIORITY 1: Check if it's an Autodoc Group domain (from SKINS)
            if domain in OWN_DOMAINS:
                return 'Autodoc Group'

            # PRIORITY 2: Check for explicit Autodoc patterns (backup)
            if 'autodoc' in domain_clean or 'auto-doc' in domain_clean:
                return 'Autodoc Group'

            # PRIORITY 3: Check for Ridex brand (separate from Autodoc, but REXBO skin belongs to Autodoc!)
            # Exception: REXBO skin domains should be Autodoc Group, not Ridex
            if 'ridex' in domain_clean:
                return 'Ridex'
            elif 'rexbo' in domain_clean:
                # Special case: REXBO domains are part of SKINS, so should be Autodoc Group
                # But only if they're actually in our SKINS structure
                rexbo_domains = SKINS.get('REXBO', {}).values()
                if domain in rexbo_domains:
                    return 'Autodoc Group'
                else:
                    return 'Ridex'  # External REXBO domains

            try:
                # PRIORITY 4: Extract brand from any other domain
                # Remove protocol if present
                if domain_clean.startswith(('http://', 'https://')):
                    domain_clean = domain_clean.split('//', 1)[1]

                # Remove common subdomains
                subdomains_to_remove = ['www.', 'shop.', 'store.', 'online.', 'web.', 'm.', 'mobile.']
                for subdomain in subdomains_to_remove:
                    if domain_clean.startswith(subdomain):
                        domain_clean = domain_clean[len(subdomain):]
                        break

                # Split by dots and get main domain part
                domain_parts = domain_clean.split('.')
                if len(domain_parts) >= 2:
                    # Get the main domain part (before TLD)
                    main_domain = domain_parts[0]

                    # Handle special domain patterns
                    if main_domain == 'fair-garage':
                        return 'Fair Garage'
                    elif main_domain == 'auto-doc':
                        return 'Autodoc Group'
                    elif '-' in main_domain:
                        # For hyphenated domains, try to extract the main brand
                        parts = main_domain.split('-')
                        if len(parts) >= 2:
                            # Take the first significant part
                            main_part = parts[0]
                            if len(main_part) > 2:
                                return main_part.title()
                            # If first part is too short, try combining
                            return '-'.join(parts[:2]).title()

                    # Clean and return the main domain
                    if main_domain and len(main_domain) > 1:
                        # Remove common suffixes that aren't brand names
                        cleaned = main_domain.replace('24', '').replace('shop', '').replace('store', '').strip('-').strip('_')

                        # If nothing left after cleaning, use original
                        if not cleaned or len(cleaned) <= 1:
                            cleaned = main_domain

                        return cleaned.title()

                # Fallback: try to extract something meaningful
                fallback = domain_clean.split('.')[0].strip()
                if fallback and len(fallback) > 1:
                    return fallback.title()

                return 'Unknown'

            except Exception as e:
                # Ultimate fallback
                try:
                    simple = domain_clean.split('.')[0]
                    return simple.title() if simple and len(simple) > 1 else 'Unknown'
                except:
                    return 'Unknown'

        # Apply brand classification
        base_occurrences['Brand'] = base_occurrences['Domain'].apply(classify_brand)

        # Create brand family grouping for high-level analysis
        def create_brand_family(brand):
            """Group brands into families for executive summary"""
            if brand == 'Autodoc Group':
                return 'Autodoc Group'
            elif brand == 'Ridex':
                return 'Ridex'
            else:
                return 'Competitors'

        base_occurrences['Brand_Family'] = base_occurrences['Brand'].apply(create_brand_family)


        # Show filter summary
        active_filters = []
        if f_country:
            country_names = {"de": "Germany", "fr": "France", "es": "Spain", "gb": "UK", "it": "Italy"}
            countries_display = [country_names.get(c, c.upper()) for c in f_country]
            active_filters.append(f"Country: {', '.join(countries_display)}")
        if f_cat:
            active_filters.append(f"Category: {', '.join(f_cat)}")
        if f_int:
            active_filters.append(f"Intent: {', '.join(f_int)}")
        if f_route:
            active_filters.append(f"Route: {', '.join(f_route)}")
        if f_skins:
            active_filters.append(f"Skins: {', '.join(f_skins)}")
        if dom_sel:
            active_filters.append(f"Additional Domains: {', '.join(dom_sel[:3])}{'...' if len(dom_sel) > 3 else ''}")
        if sel_terms:
            active_filters.append(f"Search Terms: {', '.join(sel_terms[:2])}{'...' if len(sel_terms) > 2 else ''}")

        if active_filters:
            st.info(f"📊 **Active Filters**: {' | '.join(active_filters)}")

        # DEBUG SECTION - expandable
        with st.expander("🔍 Brand Classification Debug", expanded=False):
            st.markdown("**SKINS Structure (All these domains → Autodoc Group):**")
            for skin_name, skin_domains in SKINS.items():
                st.write(f"- **{skin_name}**: {', '.join(skin_domains.values())}")

            st.markdown("**Top 20 Domains in Dataset:**")
            top_domains = base_occurrences['Domain'].value_counts().head(20)
            for domain, count in top_domains.items():
                brand = classify_brand(domain)
                brand_family = create_brand_family(brand)
                is_own = "✓" if domain in OWN_DOMAINS else "✗"
                st.write(f"- `{domain}` → **{brand}** ({brand_family}) [{is_own} Own Domain] ({count} appearances)")

            st.markdown("**Brand Distribution (Specific Brands):**")
            brand_counts = base_occurrences['Brand'].value_counts()
            for brand, count in brand_counts.items():
                percentage = round((count / len(base_occurrences) * 100), 1)
                st.write(f"- **{brand}**: {count} appearances ({percentage}%)")

            st.markdown("**Brand Group Distribution (High-Level):**")
            family_counts = base_occurrences['Brand_Family'].value_counts()
            for family, count in family_counts.items():
                percentage = round((count / len(base_occurrences) * 100), 1)
                st.write(f"- **{family}**: {count} appearances ({percentage}%)")

            st.markdown("**OWN_DOMAINS List (All SKINS domains):**")
            st.write(f"Total: {len(OWN_DOMAINS)} domains")
            st.write(sorted(list(OWN_DOMAINS)))

            st.markdown("**Classification Rules:**")
            st.write("""
            **🏢 Autodoc Group**:
            - ✅ ALL domains in SKINS structure (including REXBO skin!)
            - ✅ Explicit 'autodoc' or 'auto-doc' patterns

            **🎭 Ridex**:
            - ✅ External domains containing 'ridex'
            - ❌ REXBO skin domains → Autodoc Group (not Ridex!)

            **🏪 All Other Brands**:
            - Extract main domain name (remove subdomains, TLD, suffixes)
            - Examples: shop.ford.de → Ford, amazon.de → Amazon, fair-garage.com → Fair-Garage
            """)

        # === 1. OVERALL MARKET SHARE ===
        st.subheader("🏆 AI Visibility Market Share")

        # Calculate brand visibility scores (specific brands)
        brand_performance = base_occurrences.groupby('Brand').agg(
            Total_Visibility=('Weight', 'sum'),
            Total_Appearances=('Position', 'count'),
            Avg_Position=('Position', 'mean'),
            Unique_Search_Terms=('Search_Term', 'nunique'),
            Unique_Domains=('Domain', 'nunique')
        ).round(3)

        # Calculate market share percentages
        total_market_visibility = brand_performance['Total_Visibility'].sum()
        brand_performance['Market_Share_%'] = (brand_performance['Total_Visibility'] / total_market_visibility * 100).round(1)

        # Sort by market share
        brand_performance = brand_performance.sort_values('Market_Share_%', ascending=False)

        # Calculate family-level summaries for metrics
        family_summary = base_occurrences.groupby('Brand_Family').agg(
            Family_Visibility=('Weight', 'sum')
        )
        family_total_visibility = family_summary['Family_Visibility'].sum()
        family_summary['Family_Share_%'] = (family_summary['Family_Visibility'] / family_total_visibility * 100).round(1)

        # Display high-level family metrics
        col1, col2, col3 = st.columns(3)

        autodoc_share = family_summary.loc['Autodoc Group', 'Family_Share_%'] if 'Autodoc Group' in family_summary.index else 0
        ridex_share = family_summary.loc['Ridex', 'Family_Share_%'] if 'Ridex' in family_summary.index else 0
        competitor_share = family_summary.loc['Competitors', 'Family_Share_%'] if 'Competitors' in family_summary.index else 0

        with col1:
            st.metric("🏢 Autodoc Group Market Share", f"{autodoc_share:.1f}%",
                     help="Includes ALL SKINS domains + REXBO (but not external Ridex)")
        with col2:
            st.metric("🎭 Ridex Brand Market Share", f"{ridex_share:.1f}%",
                     help="External Ridex domains only (REXBO skin → Autodoc Group)")
        with col3:
            st.metric("⚔️ All Competitors Market Share", f"{competitor_share:.1f}%",
                     help="All external competitors combined")

        # Detailed brand performance table with filters
        st.markdown("#### 📊 Brand Performance Analysis")

        # Add filter controls
        col1, col2 = st.columns(2)
        with col1:
            show_top_n = st.selectbox("Show Top N Brands", [5, 10, 15, 20, 25, 50, "All"], index=2, key="brand_filter")
        with col2:
            sort_by = st.selectbox("Sort by", ["Market_Share_%", "Total_Visibility", "Total_Appearances", "Avg_Position"], index=0, key="sort_filter")

        # Apply filters
        if show_top_n == "All":
            displayed_brands = brand_performance
        else:
            displayed_brands = brand_performance.head(show_top_n)

        # Sort data
        if sort_by == "Avg_Position":
            displayed_brands = displayed_brands.sort_values(sort_by, ascending=True)  # Lower position is better
        else:
            displayed_brands = displayed_brands.sort_values(sort_by, ascending=False)

        st.dataframe(displayed_brands, use_container_width=True)

        # Market concentration analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            hhi = ((brand_performance['Market_Share_%'] ** 2).sum() / 10000)  # Normalize HHI
            concentration_level = "Highly Concentrated" if hhi > 2500 else "Moderately Concentrated" if hhi > 1500 else "Competitive"
            st.metric("📊 Market Concentration (HHI)", f"{hhi:.0f}", help=f"Market is {concentration_level}")

        with col2:
            top3_share = brand_performance.head(3)['Market_Share_%'].sum()
            st.metric("🏆 Top 3 Market Share", f"{top3_share:.1f}%", help="Combined market share of top 3 brands")

        with col3:
            total_brands_count = len(brand_performance)
            st.metric("🏢 Total Active Brands", total_brands_count, help="Number of brands with market presence")

        # Brand market share visualization with filters
        if not brand_performance.empty:
            # Get filtered data for pie chart
            chart_top_n = st.selectbox("Pie Chart: Show Top N Brands", [5, 10, 15, 20], index=1, key="pie_filter")
            pie_data = brand_performance.head(chart_top_n).reset_index()

            # Add "Others" category if there are more brands
            if len(brand_performance) > chart_top_n:
                others_share = brand_performance.iloc[chart_top_n:]['Market_Share_%'].sum()
                others_row = pd.DataFrame({
                    'Brand': ['Others'],
                    'Market_Share_%': [others_share],
                    'Total_Visibility': [brand_performance.iloc[chart_top_n:]['Total_Visibility'].sum()],
                    'Total_Appearances': [brand_performance.iloc[chart_top_n:]['Total_Appearances'].sum()]
                })
                pie_data = pd.concat([pie_data, others_row], ignore_index=True)

            # Create pie chart with dynamic data
            pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Market_Share_%", type="quantitative"),
                color=alt.Color(field="Brand", type="nominal",
                               scale=alt.Scale(scheme='category20')),
                tooltip=['Brand:N', 'Market_Share_%:Q', 'Total_Visibility:Q', 'Total_Appearances:Q']
            ).properties(height=400, title=f"Market Share - Top {chart_top_n} Brands + Others")

            st.altair_chart(pie_chart, use_container_width=True)

            # Horizontal bar chart for better readability
            bar_chart = alt.Chart(pie_data.head(15)).mark_bar().encode(
                x=alt.X('Market_Share_%:Q', title='Market Share %'),
                y=alt.Y('Brand:N', sort=alt.EncodingSortField(field='Market_Share_%', order='descending')),
                color=alt.Color('Brand:N', legend=None, scale=alt.Scale(scheme='viridis')),
                tooltip=['Brand:N', 'Market_Share_%:Q', 'Total_Visibility:Q', 'Total_Appearances:Q']
            ).properties(height=500, title="Top 15 Brands - Market Share Comparison")

            st.altair_chart(bar_chart, use_container_width=True)

        # === 2. GEOGRAPHIC MARKET ANALYSIS ===
        st.subheader("🌍 Geographic Market Distribution")

        # High-level family analysis by country
        country_family_analysis = base_occurrences.groupby(['Lang', 'Brand_Family']).agg(
            Visibility=('Weight', 'sum'),
            Appearances=('Position', 'count')
        ).reset_index()

        # Calculate country market shares for families
        country_totals = country_family_analysis.groupby('Lang')['Visibility'].sum()
        country_family_analysis['Country_Market_Share_%'] = country_family_analysis.apply(
            lambda row: (row['Visibility'] / country_totals[row['Lang']] * 100).round(1), axis=1
        )

        # Pivot for better display
        country_family_pivot = country_family_analysis.pivot(index='Lang', columns='Brand_Family', values='Country_Market_Share_%').fillna(0)

        st.markdown("#### 🗺️ Market Share by Country (Brand Groups)")
        st.dataframe(country_family_pivot, use_container_width=True)

        # Detailed brand analysis by country (top brands only)
        st.markdown("#### 🎯 Top Competitor Brands by Country")
        for country in base_occurrences['Lang'].unique():
            country_data = base_occurrences[base_occurrences['Lang'] == country]
            country_brands = country_data.groupby('Brand').agg(
                Country_Visibility=('Weight', 'sum')
            ).sort_values('Country_Visibility', ascending=False).head(5)

            country_total = country_brands['Country_Visibility'].sum()
            country_brands['Country_Share_%'] = (country_brands['Country_Visibility'] / country_total * 100).round(1)

            with st.expander(f"🏴 {country.upper()} - Top 5 Brands", expanded=False):
                st.dataframe(country_brands, use_container_width=True)

        # Geographic performance chart (families)
        if not country_family_analysis.empty:
            geo_chart = alt.Chart(country_family_analysis).mark_bar().encode(
                x=alt.X('Lang:N', title='Country'),
                y=alt.Y('Country_Market_Share_%:Q', title='Market Share %'),
                color=alt.Color('Brand_Family:N',
                               scale=alt.Scale(domain=['Autodoc Group', 'Ridex', 'Competitors'],
                                             range=['#2E86AB', '#A23B72', '#F24236'])),
                tooltip=['Lang:N', 'Brand_Family:N', 'Country_Market_Share_%:Q', 'Visibility:Q']
            ).properties(height=400, title="Market Share by Country and Brand Group")

            st.altair_chart(geo_chart, use_container_width=True)

        # === 3. ADVANCED TIME-BASED TREND ANALYSIS ===
        st.subheader("📈 Advanced Market Trend Analysis - Weekly & Monthly")

        # Check if we have sufficient time data for trends
        unique_dates = base_occurrences['Date'].unique()
        if len(unique_dates) > 1:
            # Sort dates for proper analysis
            sorted_dates = sorted(unique_dates)

            # === MONDAY-ONLY ANALYSIS (Future-proof for weekly data) ===
            st.markdown("#### 📅 **Monday-Only Weekly Analysis**")

            def get_monday_data(df):
                """Filter data to only Monday entries for consistent weekly comparison"""
                df_with_weekday = df.copy()
                df_with_weekday['Weekday'] = pd.to_datetime(df_with_weekday['Date']).dt.day_name()
                monday_data = df_with_weekday[df_with_weekday['Weekday'] == 'Monday']

                if monday_data.empty:
                    # If no Monday data, use the earliest date of each week
                    df_with_weekday['Week'] = pd.to_datetime(df_with_weekday['Date']).dt.isocalendar().week
                    df_with_weekday['Year'] = pd.to_datetime(df_with_weekday['Date']).dt.year

                    # Get earliest date per week
                    weekly_earliest = df_with_weekday.groupby(['Year', 'Week'])['Date'].min().reset_index()
                    monday_data = df_with_weekday[df_with_weekday['Date'].isin(weekly_earliest['Date'])]

                return monday_data

            # Get Monday-only data for analysis
            monday_base_data = get_monday_data(base_data)
            monday_base_occurrences = get_monday_data(base_occurrences)

            # Show Monday data availability
            monday_dates = sorted(monday_base_data['Date'].unique()) if not monday_base_data.empty else []

            if len(monday_dates) >= 2:
                st.success(f"📊 **Monday Analysis Available**: {len(monday_dates)} Monday datasets found")
                st.info(f"📅 **Analysis Dates**: {', '.join([d.strftime('%Y-%m-%d') for d in monday_dates])}")

                # === WEEK-OVER-WEEK ANALYSIS ===
                st.markdown("##### 📈 **Week-over-Week Market Share Changes**")

                weekly_analysis_results = []

                for i in range(1, len(monday_dates)):
                    current_week = monday_dates[i]
                    previous_week = monday_dates[i-1]

                    # Get data for current and previous week
                    current_data = monday_base_occurrences[monday_base_occurrences['Date'] == current_week]
                    previous_data = monday_base_occurrences[monday_base_occurrences['Date'] == previous_week]

                    if not current_data.empty and not previous_data.empty:
                        # Calculate weekly market shares
                        current_brands = current_data.groupby('Brand_Family')['Weight'].sum()
                        previous_brands = previous_data.groupby('Brand_Family')['Weight'].sum()

                        current_total = current_brands.sum()
                        previous_total = previous_brands.sum()

                        current_shares = (current_brands / current_total * 100).round(1)
                        previous_shares = (previous_brands / previous_total * 100).round(1)

                        # Calculate changes
                        for brand_family in ['Autodoc Group', 'Ridex', 'Competitors']:
                            current_share = current_shares.get(brand_family, 0)
                            previous_share = previous_shares.get(brand_family, 0)
                            change_points = current_share - previous_share

                            weekly_analysis_results.append({
                                'Week_Ending': current_week.strftime('%Y-%m-%d'),
                                'Brand_Family': brand_family,
                                'Current_Share_%': current_share,
                                'Previous_Share_%': previous_share,
                                'Change_Points': change_points,
                                'Change_%': ((current_share - previous_share) / previous_share * 100) if previous_share > 0 else 0
                            })

                if weekly_analysis_results:
                    weekly_df = pd.DataFrame(weekly_analysis_results)

                    # Display latest week-over-week changes
                    latest_week = weekly_df['Week_Ending'].max()
                    latest_changes = weekly_df[weekly_df['Week_Ending'] == latest_week]

                    if not latest_changes.empty:
                        st.markdown(f"**📊 Latest Week-over-Week Changes (Week ending {latest_week}):**")

                        col1, col2, col3 = st.columns(3)

                        for i, (_, row) in enumerate(latest_changes.iterrows()):
                            brand_family = row['Brand_Family']
                            change_points = row['Change_Points']
                            current_share = row['Current_Share_%']

                            # Choose column
                            with [col1, col2, col3][i % 3]:
                                change_indicator = "📈" if change_points > 0 else "📉" if change_points < 0 else "➡️"
                                st.metric(
                                    f"{change_indicator} {brand_family}",
                                    f"{current_share:.1f}%",
                                    f"{change_points:+.1f} pts WoW",
                                    delta_color="normal" if change_points >= 0 else "inverse"
                                )

                    # === WEEKLY TREND VISUALIZATION ===
                    st.markdown("##### 📊 **Weekly Market Share Trend Chart**")

                    if not weekly_df.empty:
                        weekly_chart = alt.Chart(weekly_df).mark_line(point=True, strokeWidth=3).encode(
                            x=alt.X('Week_Ending:T', title='Week Ending'),
                            y=alt.Y('Current_Share_%:Q', title='Market Share %'),
                            color=alt.Color('Brand_Family:N',
                                           scale=alt.Scale(domain=['Autodoc Group', 'Ridex', 'Competitors'],
                                                         range=['#2E86AB', '#A23B72', '#F24236'])),
                            tooltip=['Week_Ending:T', 'Brand_Family:N', 'Current_Share_%:Q', 'Change_Points:Q']
                        ).properties(height=400, title="Weekly Market Share Trends (Monday Data Only)")

                        st.altair_chart(weekly_chart, use_container_width=True)

                    # === WEEKLY CHANGES TABLE ===
                    st.markdown("##### 📋 **Complete Weekly Changes Analysis**")

                    # Pivot table for better readability
                    weekly_pivot = weekly_df.pivot(index='Week_Ending', columns='Brand_Family', values='Change_Points').round(1)
                    weekly_pivot.columns = [f"{col}_Change_Points" for col in weekly_pivot.columns]

                    # Add current shares for context
                    weekly_shares_pivot = weekly_df.pivot(index='Week_Ending', columns='Brand_Family', values='Current_Share_%').round(1)
                    weekly_shares_pivot.columns = [f"{col}_Share_%" for col in weekly_shares_pivot.columns]

                    # Combine both tables
                    combined_weekly = pd.concat([weekly_shares_pivot, weekly_pivot], axis=1)

                    st.dataframe(combined_weekly, use_container_width=True)

                    # Download weekly analysis
                    st.download_button(
                        "📥 Download Weekly Analysis",
                        weekly_df.to_csv(index=False).encode("utf-8"),
                        "weekly_market_analysis.csv",
                        "text/csv",
                        key="dl_weekly_analysis"
                    )

                # === MONTH-OVER-MONTH ANALYSIS ===
                if len(monday_dates) >= 4:  # Need at least 4 weeks for monthly comparison
                    st.markdown("##### 📅 **Month-over-Month Analysis**")

                    # Group dates by month
                    monthly_data = {}
                    for date in monday_dates:
                        month_key = date.strftime('%Y-%m')
                        if month_key not in monthly_data:
                            monthly_data[month_key] = []
                        monthly_data[month_key].append(date)

                    # Calculate monthly aggregates (using latest Monday of each month)
                    monthly_results = []
                    months = sorted(monthly_data.keys())

                    for month in months:
                        # Use latest Monday of the month
                        latest_monday = max(monthly_data[month])
                        month_data = monday_base_occurrences[monday_base_occurrences['Date'] == latest_monday]

                        if not month_data.empty:
                            month_brands = month_data.groupby('Brand_Family')['Weight'].sum()
                            month_total = month_brands.sum()
                            month_shares = (month_brands / month_total * 100).round(1)

                            for brand_family in ['Autodoc Group', 'Ridex', 'Competitors']:
                                monthly_results.append({
                                    'Month': month,
                                    'Brand_Family': brand_family,
                                    'Month_Share_%': month_shares.get(brand_family, 0),
                                    'Representative_Date': latest_monday.strftime('%Y-%m-%d')
                                })

                    if len(monthly_results) >= 6:  # At least 2 months of data for comparison
                        monthly_df = pd.DataFrame(monthly_results)

                        # Calculate month-over-month changes
                        monthly_changes = []
                        unique_months = sorted(monthly_df['Month'].unique())

                        for i in range(1, len(unique_months)):
                            current_month = unique_months[i]
                            previous_month = unique_months[i-1]

                            current_month_data = monthly_df[monthly_df['Month'] == current_month]
                            previous_month_data = monthly_df[monthly_df['Month'] == previous_month]

                            for brand_family in ['Autodoc Group', 'Ridex', 'Competitors']:
                                current_share = current_month_data[current_month_data['Brand_Family'] == brand_family]['Month_Share_%'].iloc[0] if not current_month_data[current_month_data['Brand_Family'] == brand_family].empty else 0
                                previous_share = previous_month_data[previous_month_data['Brand_Family'] == brand_family]['Month_Share_%'].iloc[0] if not previous_month_data[previous_month_data['Brand_Family'] == brand_family].empty else 0

                                change_points = current_share - previous_share

                                monthly_changes.append({
                                    'Month': current_month,
                                    'Brand_Family': brand_family,
                                    'Current_Month_Share_%': current_share,
                                    'Previous_Month_Share_%': previous_share,
                                    'MoM_Change_Points': change_points,
                                    'MoM_Change_%': ((current_share - previous_share) / previous_share * 100) if previous_share > 0 else 0
                                })

                        if monthly_changes:
                            monthly_changes_df = pd.DataFrame(monthly_changes)

                            # Display latest month-over-month changes
                            latest_month = monthly_changes_df['Month'].max()
                            latest_monthly_changes = monthly_changes_df[monthly_changes_df['Month'] == latest_month]

                            st.markdown(f"**📅 Latest Month-over-Month Changes ({latest_month}):**")

                            col1, col2, col3 = st.columns(3)

                            for i, (_, row) in enumerate(latest_monthly_changes.iterrows()):
                                brand_family = row['Brand_Family']
                                change_points = row['MoM_Change_Points']
                                current_share = row['Current_Month_Share_%']

                                with [col1, col2, col3][i % 3]:
                                    change_indicator = "📈" if change_points > 0 else "📉" if change_points < 0 else "➡️"
                                    st.metric(
                                        f"{change_indicator} {brand_family}",
                                        f"{current_share:.1f}%",
                                        f"{change_points:+.1f} pts MoM",
                                        delta_color="normal" if change_points >= 0 else "inverse"
                                    )

                            # Monthly trend visualization
                            monthly_chart = alt.Chart(monthly_df).mark_line(point=True, strokeWidth=3).encode(
                                x=alt.X('Month:T', title='Month'),
                                y=alt.Y('Month_Share_%:Q', title='Market Share %'),
                                color=alt.Color('Brand_Family:N',
                                               scale=alt.Scale(domain=['Autodoc Group', 'Ridex', 'Competitors'],
                                                             range=['#2E86AB', '#A23B72', '#F24236'])),
                                tooltip=['Month:T', 'Brand_Family:N', 'Month_Share_%:Q', 'Representative_Date:N']
                            ).properties(height=400, title="Monthly Market Share Trends")

                            st.altair_chart(monthly_chart, use_container_width=True)

                            # Monthly changes table
                            st.markdown("##### 📋 **Monthly Changes Analysis**")
                            monthly_pivot = monthly_changes_df.pivot(index='Month', columns='Brand_Family', values='MoM_Change_Points').round(1)
                            st.dataframe(monthly_pivot, use_container_width=True)

                            # Download monthly analysis
                            st.download_button(
                                "📥 Download Monthly Analysis",
                                monthly_changes_df.to_csv(index=False).encode("utf-8"),
                                "monthly_market_analysis.csv",
                                "text/csv",
                                key="dl_monthly_analysis"
                            )
                    else:
                        st.info("📅 Monthly analysis requires at least 2 months of data.")
                else:
                    st.info("📅 Monthly analysis requires at least 4 weeks of data.")

            else:
                st.warning("⚠️ No Monday data available for weekly analysis. Using available dates for trend analysis.")

            # === BRAND-SPECIFIC WEEKLY PERFORMANCE ===
            if len(monday_dates) >= 2:
                st.markdown("#### 🎯 **Brand-Specific Weekly Performance**")

                # Allow selection of specific brands to track
                available_brands = brand_performance.head(10).index.tolist()  # Top 10 brands
                selected_brands_for_trends = st.multiselect(
                    "Select brands for detailed weekly tracking:",
                    available_brands,
                    default=available_brands[:5],  # Default to top 5
                    key="brand_weekly_trends"
                )

                if selected_brands_for_trends:
                    brand_weekly_data = []

                    for date in monday_dates:
                        date_data = monday_base_occurrences[monday_base_occurrences['Date'] == date]
                        if not date_data.empty:
                            date_brand_performance = date_data.groupby('Brand')['Weight'].sum()
                            total_visibility = date_brand_performance.sum()

                            for brand in selected_brands_for_trends:
                                brand_visibility = date_brand_performance.get(brand, 0)
                                brand_share = (brand_visibility / total_visibility * 100) if total_visibility > 0 else 0

                                brand_weekly_data.append({
                                    'Date': date,
                                    'Brand': brand,
                                    'Market_Share_%': brand_share,
                                    'Visibility_Score': brand_visibility
                                })

                    if brand_weekly_data:
                        brand_weekly_df = pd.DataFrame(brand_weekly_data)

                        # Brand-specific trend chart
                        brand_weekly_chart = alt.Chart(brand_weekly_df).mark_line(point=True).encode(
                            x=alt.X('Date:T', title='Date'),
                            y=alt.Y('Market_Share_%:Q', title='Market Share %'),
                            color=alt.Color('Brand:N', scale=alt.Scale(scheme='category20')),
                            tooltip=['Date:T', 'Brand:N', 'Market_Share_%:Q', 'Visibility_Score:Q']
                        ).properties(height=400, title="Brand-Specific Weekly Market Share Trends")

                        st.altair_chart(brand_weekly_chart, use_container_width=True)

                        # Latest week performance for selected brands
                        latest_week_data = brand_weekly_df[brand_weekly_df['Date'] == brand_weekly_df['Date'].max()]
                        if len(monday_dates) >= 2:
                            previous_week_data = brand_weekly_df[brand_weekly_df['Date'] == sorted(brand_weekly_df['Date'].unique())[-2]]

                            # Calculate week-over-week changes for each brand
                            brand_changes = []
                            for brand in selected_brands_for_trends:
                                current_share = latest_week_data[latest_week_data['Brand'] == brand]['Market_Share_%'].iloc[0] if not latest_week_data[latest_week_data['Brand'] == brand].empty else 0
                                previous_share = previous_week_data[previous_week_data['Brand'] == brand]['Market_Share_%'].iloc[0] if not previous_week_data[previous_week_data['Brand'] == brand].empty else 0

                                change = current_share - previous_share
                                brand_changes.append({
                                    'Brand': brand,
                                    'Current_Share_%': current_share,
                                    'Previous_Share_%': previous_share,
                                    'WoW_Change_Points': change
                                })

                            brand_changes_df = pd.DataFrame(brand_changes).sort_values('WoW_Change_Points', ascending=False)

                            st.markdown("##### 🏆 **Brand Weekly Winners & Losers**")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📈 Top Weekly Gainers:**")
                                top_gainers = brand_changes_df.head(3)
                                for _, row in top_gainers.iterrows():
                                    if row['WoW_Change_Points'] > 0:
                                        st.success(f"**{row['Brand']}**: +{row['WoW_Change_Points']:.1f} pts ({row['Current_Share_%']:.1f}%)")

                            with col2:
                                st.markdown("**📉 Weekly Decliners:**")
                                top_losers = brand_changes_df.tail(3)
                                for _, row in top_losers.iterrows():
                                    if row['WoW_Change_Points'] < 0:
                                        st.error(f"**{row['Brand']}**: {row['WoW_Change_Points']:.1f} pts ({row['Current_Share_%']:.1f}%)")

            # === PERFORMANCE CHANGES ===
            st.markdown("#### 🔄 Market Performance Changes")

            # Calculate Week-over-Week and Month-over-Month if possible
            def calculate_performance_changes():
                """Calculate professional market performance metrics"""
                results = {}

                # Get latest and previous periods
                if len(sorted_dates) >= 2:
                    latest_date = sorted_dates[-1]
                    previous_date = sorted_dates[-2]

                    # Current period data
                    current_data = base_occurrences[base_occurrences['Date'] == latest_date].groupby('Brand').agg(
                        Current_Visibility=('Weight', 'sum'),
                        Current_Appearances=('Position', 'count')
                    )

                    # Previous period data
                    previous_data = base_occurrences[base_occurrences['Date'] == previous_date].groupby('Brand').agg(
                        Previous_Visibility=('Weight', 'sum'),
                        Previous_Appearances=('Position', 'count')
                    )

                    # Merge data for comparison
                    comparison_data = current_data.join(previous_data, how='outer').fillna(0)

                    # Calculate changes
                    comparison_data['Visibility_Change_%'] = (
                        (comparison_data['Current_Visibility'] - comparison_data['Previous_Visibility']) /
                        comparison_data['Previous_Visibility'].replace(0, 1) * 100
                    ).round(1)

                    comparison_data['Appearances_Change_%'] = (
                        (comparison_data['Current_Appearances'] - comparison_data['Previous_Appearances']) /
                        comparison_data['Previous_Appearances'].replace(0, 1) * 100
                    ).round(1)

                    # Calculate market share changes
                    current_total = comparison_data['Current_Visibility'].sum()
                    previous_total = comparison_data['Previous_Visibility'].sum()

                    comparison_data['Current_Market_Share_%'] = (comparison_data['Current_Visibility'] / current_total * 100).round(2)
                    comparison_data['Previous_Market_Share_%'] = (comparison_data['Previous_Visibility'] / previous_total * 100).round(2)
                    comparison_data['Market_Share_Change_Points'] = (comparison_data['Current_Market_Share_%'] - comparison_data['Previous_Market_Share_%']).round(2)

                    # Filter significant changes only
                    comparison_data = comparison_data[
                        (comparison_data['Current_Visibility'] > 0) | (comparison_data['Previous_Visibility'] > 0)
                    ]

                    return comparison_data.sort_values('Market_Share_Change_Points', ascending=False)

                return pd.DataFrame()

            performance_changes = calculate_performance_changes()

            if not performance_changes.empty:
                # === TOP GAINERS & LOSERS ===
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 📈 **Top 5 Market Share Gainers**")
                    top_gainers = performance_changes.head(5)[['Current_Market_Share_%', 'Market_Share_Change_Points', 'Visibility_Change_%']]

                    # Style the dataframe for gainers
                    def style_gainers(val):
                        if isinstance(val, (int, float)):
                            if val > 0:
                                return 'color: green; font-weight: bold'
                        return ''

                    st.dataframe(top_gainers.style.applymap(style_gainers), use_container_width=True)

                with col2:
                    st.markdown("##### 📉 **Top 5 Market Share Losers**")
                    top_losers = performance_changes.tail(5)[['Current_Market_Share_%', 'Market_Share_Change_Points', 'Visibility_Change_%']]

                    # Style the dataframe for losers
                    def style_losers(val):
                        if isinstance(val, (int, float)):
                            if val < 0:
                                return 'color: red; font-weight: bold'
                        return ''

                    st.dataframe(top_losers.style.applymap(style_losers), use_container_width=True)

                # === BRAND FAMILY CHANGES ===
                st.markdown("##### 🏢 **Brand Group Performance Changes**")

                # Calculate family-level changes
                family_current = base_occurrences[base_occurrences['Date'] == sorted_dates[-1]].groupby('Brand_Family').agg(
                    Current_Family_Visibility=('Weight', 'sum')
                )
                family_previous = base_occurrences[base_occurrences['Date'] == sorted_dates[-2]].groupby('Brand_Family').agg(
                    Previous_Family_Visibility=('Weight', 'sum')
                )

                family_comparison = family_current.join(family_previous, how='outer').fillna(0)
                family_total_current = family_comparison['Current_Family_Visibility'].sum()
                family_total_previous = family_comparison['Previous_Family_Visibility'].sum()

                family_comparison['Current_Family_Share_%'] = (family_comparison['Current_Family_Visibility'] / family_total_current * 100).round(1)
                family_comparison['Previous_Family_Share_%'] = (family_comparison['Previous_Family_Visibility'] / family_total_previous * 100).round(1)
                family_comparison['Family_Change_Points'] = (family_comparison['Current_Family_Share_%'] - family_comparison['Previous_Family_Share_%']).round(1)

                col1, col2, col3 = st.columns(3)

                for i, (brand_family, row) in enumerate(family_comparison.iterrows()):
                    change_color = "🟢" if row['Family_Change_Points'] > 0 else "🔴" if row['Family_Change_Points'] < 0 else "⚪"

                    with [col1, col2, col3][i % 3]:
                        st.metric(
                            f"{change_color} {brand_family}",
                            f"{row['Current_Family_Share_%']:.1f}%",
                            f"{row['Family_Change_Points']:+.1f} pts vs previous period",
                            delta_color="normal" if row['Family_Change_Points'] >= 0 else "inverse"
                        )

                # === DETAILED PERFORMANCE TABLE ===
                st.markdown("##### 📋 **Complete Brand Performance Changes**")

                # Show all brands with changes (filterable)
                change_filter = st.selectbox("Show brands with:",
                                           ["All Changes", "Positive Changes Only", "Negative Changes Only", "Significant Changes (>1%)"],
                                           key="change_filter")

                if change_filter == "Positive Changes Only":
                    filtered_changes = performance_changes[performance_changes['Market_Share_Change_Points'] > 0]
                elif change_filter == "Negative Changes Only":
                    filtered_changes = performance_changes[performance_changes['Market_Share_Change_Points'] < 0]
                elif change_filter == "Significant Changes (>1%)":
                    filtered_changes = performance_changes[abs(performance_changes['Market_Share_Change_Points']) > 1.0]
                else:
                    filtered_changes = performance_changes

                # Format the changes table for better readability
                display_changes = filtered_changes[[
                    'Current_Market_Share_%', 'Previous_Market_Share_%', 'Market_Share_Change_Points',
                    'Current_Visibility', 'Previous_Visibility', 'Visibility_Change_%'
                ]].copy()

                st.dataframe(display_changes, use_container_width=True)

            else:
                st.info("📅 Performance change analysis requires data from at least 2 time periods.")

            # === TREND VISUALIZATION ===
            st.markdown("#### 📊 **Market Share Trends Over Time**")

            # Time-based analysis (family level for trends)
            time_analysis = base_occurrences.groupby(['Date', 'Brand_Family']).agg(
                Daily_Visibility=('Weight', 'sum'),
                Daily_Appearances=('Position', 'count')
            ).reset_index()

            # Calculate daily market shares
            daily_totals = time_analysis.groupby('Date')['Daily_Visibility'].sum()
            time_analysis['Daily_Market_Share_%'] = time_analysis.apply(
                lambda row: (row['Daily_Visibility'] / daily_totals[row['Date']] * 100).round(1), axis=1
            )

            # Brand group trend visualization
            trend_chart = alt.Chart(time_analysis).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Daily_Market_Share_%:Q', title='Market Share %'),
                color=alt.Color('Brand_Family:N',
                               scale=alt.Scale(domain=['Autodoc Group', 'Ridex', 'Competitors'],
                                             range=['#2E86AB', '#A23B72', '#F24236'])),
                tooltip=['Date:T', 'Brand_Family:N', 'Daily_Market_Share_%:Q', 'Daily_Visibility:Q']
            ).properties(height=400, title="Brand Group Market Share Trends")

            st.altair_chart(trend_chart, use_container_width=True)

            # Individual brand trend analysis for top competitors
            st.markdown("#### 🎯 **Top Brand Performance Trends**")

            trend_brands_n = st.selectbox("Show trends for top N brands:", [5, 8, 10, 15], index=1, key="trend_brands")
            top_brands_for_trends = brand_performance.head(trend_brands_n).index.tolist()

            brand_time_analysis = base_occurrences[base_occurrences['Brand'].isin(top_brands_for_trends)].groupby(['Date', 'Brand']).agg(
                Daily_Brand_Visibility=('Weight', 'sum')
            ).reset_index()

            if not brand_time_analysis.empty:
                brand_trend_chart = alt.Chart(brand_time_analysis).mark_line(point=True).encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y('Daily_Brand_Visibility:Q', title='Visibility Score'),
                    color=alt.Color('Brand:N', scale=alt.Scale(scheme='category20')),
                    tooltip=['Date:T', 'Brand:N', 'Daily_Brand_Visibility:Q']
                ).properties(height=400, title=f"Top {trend_brands_n} Brand Visibility Trends")

                st.altair_chart(brand_trend_chart, use_container_width=True)

        else:
            st.info("📅 Trend analysis requires data from multiple time periods.")

        # === 4. BRAND CITATIONS ANALYSIS ===
        st.subheader("💬 Brand Citations in AI Responses")

        # Analyze brand mentions in AI responses
        def find_brand_citations(response_text):
            """Find brand mentions in AI response text"""
            if not response_text or pd.isna(response_text):
                return []

            response_lower = str(response_text).lower()
            citations = []

            # Define brand search terms
            brand_terms = {
                'Autodoc': ['autodoc', 'auto-doc'],
                'Ridex': ['ridex', 'rexbo'],
                'Amazon': ['amazon'],
                'Norauto': ['norauto'],
                'eBay': ['ebay'],
                'kfzteile24': ['kfzteile24'],
                'teilehaber': ['teilehaber'],
                'pkwteile': ['pkwteile'],
                'Oscaro': ['oscaro'],
                'Aureliacar': ['aureliacar']
            }

            for brand, terms in brand_terms.items():
                for term in terms:
                    if term in response_lower:
                        citations.append(brand)
                        break  # Only count each brand once per response

            return citations

        # Apply citation analysis to base data
        base_data['Brand_Citations'] = base_data['Response'].apply(find_brand_citations)

        # Count citations per brand
        all_citations = []
        for idx, row in base_data.iterrows():
            if row['Brand_Citations']:  # Only if citations exist
                for brand in row['Brand_Citations']:
                    # Clean volume data - convert to numeric, handle errors
                    volume_value = 0
                    try:
                        if pd.notna(row.get('volume')):
                            volume_value = float(str(row['volume']).replace(',', '').replace('-', '0'))
                    except (ValueError, TypeError):
                        volume_value = 0

                    all_citations.append({
                        'Brand': brand,
                        'Search_Term': str(row.get('Search Term', '')),
                        'Lang': str(row.get('Lang', '')),
                        'Volume': volume_value
                    })

        if all_citations:
            citations_df = pd.DataFrame(all_citations)

            # Ensure Volume column is numeric
            citations_df['Volume'] = pd.to_numeric(citations_df['Volume'], errors='coerce').fillna(0)

            # Citation analysis
            citation_stats = citations_df.groupby('Brand').agg(
                Citation_Count=('Search_Term', 'count'),
                Unique_Search_Terms=('Search_Term', 'nunique'),
                Total_Volume=('Volume', 'sum'),
                Avg_Volume=('Volume', 'mean')
            ).round(2)

            # Calculate citation share
            total_citations = citation_stats['Citation_Count'].sum()
            citation_stats['Citation_Share_%'] = (citation_stats['Citation_Count'] / total_citations * 100).round(1)
            citation_stats = citation_stats.sort_values('Citation_Share_%', ascending=False)

            st.markdown("#### 📣 Brand Mention Analysis")
            st.dataframe(citation_stats, use_container_width=True)

            # Citation visualization
            if len(citation_stats) > 0:
                citation_chart = alt.Chart(citation_stats.reset_index()).mark_bar().encode(
                    x=alt.X('Citation_Share_%:Q', title='Citation Share %'),
                    y=alt.Y('Brand:N', sort=alt.EncodingSortField(field='Citation_Share_%', order='descending')),
                    color=alt.Color('Brand:N', legend=None),
                    tooltip=['Brand:N', 'Citation_Share_%:Q', 'Citation_Count:Q', 'Total_Volume:Q']
                ).properties(height=300, title="Brand Citations in AI Responses")

                st.altair_chart(citation_chart, use_container_width=True)

            # Citation vs URL performance comparison
            col1, col2 = st.columns(2)
            with col1:
                autodoc_citations = citation_stats.loc['Autodoc', 'Citation_Share_%'] if 'Autodoc' in citation_stats.index else 0
                st.metric("📣 Autodoc Citation Share", f"{autodoc_citations:.1f}%",
                         help="How often Autodoc is mentioned in AI responses")
            with col2:
                ridex_citations = citation_stats.loc['Ridex', 'Citation_Share_%'] if 'Ridex' in citation_stats.index else 0
                st.metric("📣 Ridex Citation Share", f"{ridex_citations:.1f}%",
                         help="How often Ridex is mentioned in AI responses")

            # Show top competitor citations
            if len(citation_stats) > 2:
                st.markdown("#### 🏆 Top Competitor Brand Citations")
                competitor_citations = citation_stats[~citation_stats.index.isin(['Autodoc', 'Ridex'])].head(5)
                if not competitor_citations.empty:
                    st.dataframe(competitor_citations[['Citation_Count', 'Citation_Share_%']], use_container_width=True)
        else:
            st.info("ℹ️ No brand citations found in current dataset.")

        # === 5. COMPETITIVE INTELLIGENCE ===
        st.subheader("🔍 Competitive Intelligence")

        # Top competitors analysis
        competitor_domains = base_occurrences[base_occurrences['Brand_Family'] == 'Competitors']
        if not competitor_domains.empty:
            top_competitors = competitor_domains.groupby('Domain').agg(
                Competitor_Visibility=('Weight', 'sum'),
                Competitor_Appearances=('Position', 'count'),
                Avg_Position=('Position', 'mean')
            ).sort_values('Competitor_Visibility', ascending=False).head(10)

            # Calculate individual competitor market shares
            total_visibility = base_occurrences['Weight'].sum()
            top_competitors['Individual_Market_Share_%'] = (top_competitors['Competitor_Visibility'] / total_visibility * 100).round(2)

            st.markdown("#### ⚔️ Top 10 Individual Competitors")
            st.dataframe(top_competitors, use_container_width=True)

            # Competitor threat assessment
            col1, col2 = st.columns(2)
            with col1:
                if not top_competitors.empty:
                    biggest_threat = top_competitors.index[0]
                    threat_share = top_competitors.iloc[0]['Individual_Market_Share_%']
                    st.metric("🚨 Biggest Competitive Threat", biggest_threat, f"{threat_share}% market share")
                else:
                    st.metric("🚨 Biggest Competitive Threat", "N/A", "No competitor data")

            with col2:
                if not top_competitors.empty:
                    avg_competitor_position = top_competitors['Avg_Position'].mean()
                    st.metric("📊 Avg Competitor Position", f"{avg_competitor_position:.1f}",
                             help="Lower is better (Position 1 = best)")
                else:
                    st.metric("📊 Avg Competitor Position", "N/A")

        # === 6. EXECUTIVE SUMMARY ===
        st.subheader("📋 Executive Summary")

        # Generate dynamic executive summary
        summary_data = {
            'autodoc_share': autodoc_share,
            'ridex_share': ridex_share,
            'competitor_share': competitor_share,
            'total_terms': base_data.shape[0],
            'total_visibility': total_market_visibility,
            'countries': len(base_occurrences['Lang'].unique()),
            'biggest_threat': top_competitors.index[0] if not competitor_domains.empty and not top_competitors.empty else "N/A",
            'threat_share': top_competitors.iloc[0]['Individual_Market_Share_%'] if not competitor_domains.empty and not top_competitors.empty else 0
        }

        executive_summary = f"""
        **AI Visibility Market Intelligence Report**

        🎯 **Market Position**: Autodoc Brand Family commands **{summary_data['autodoc_share']:.1f}%** of AI visibility market share across {summary_data['countries']} countries and {summary_data['total_terms']} search terms.

        🏆 **Brand Performance**:
        - Autodoc Family (ALL SKINS + REXBO): **{summary_data['autodoc_share']:.1f}%** market share
        - Ridex independent brand: **{summary_data['ridex_share']:.1f}%** market share
        - Combined Autodoc ecosystem: **{(summary_data['autodoc_share'] + summary_data['ridex_share']):.1f}%**

        ⚔️ **Competitive Landscape**: Main competitor "{summary_data['biggest_threat']}" holds **{summary_data['threat_share']:.1f}%** individual market share. Total competitive pressure: **{summary_data['competitor_share']:.1f}%**.

        📈 **Strategic Implications**: {"Autodoc dominates" if summary_data['autodoc_share'] > 30 else "Competitive market" if summary_data['autodoc_share'] > 20 else "Growth opportunity"} in AI search visibility. Focus on {"maintaining leadership" if summary_data['autodoc_share'] > 30 else "expanding market share" if summary_data['autodoc_share'] > 15 else "aggressive growth strategy"}.
        """

        st.markdown(executive_summary)

        # Download enhanced market analysis
        download_data = brand_performance.copy()
        download_data['Brand_Group'] = download_data.index.map(lambda x: create_brand_family(x))

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Complete Brand Analysis",
                download_data.reset_index().to_csv(index=False).encode("utf-8"),
                "complete_brand_market_analysis.csv",
                "text/csv",
                key="dl_complete_analysis"
            )

# ═════════════ TAB – N-gram Insights ═════
with tab_ngram:
    st.header("🔍 Answer N-gram Insights")

    if len(sel_terms) != 1:
        st.info("ℹ️ Please select exactly **one** Search Term in the sidebar.")
    else:
        term = sel_terms[0]
        resp_df = df_view[df_view["Search Term"] == term]
        if resp_df.empty:
            st.warning("⚠️ No responses for this term.")
        else:
            st.success(f"📊 Analyzing responses for: **{term}**")

            dom_pat = r"\b\w+\.(?:de|com|co\.uk|fr|es|it|nl|pl|cz|sk|hu|dk|fi|se|no|ch|be|bg|ro|pt|gr)\b"
            url_pat = r"http\S+"
            clean_resp = (resp_df["Response"].astype(str)
                          .str.replace(url_pat, " ", regex=True)
                          .str.replace(dom_pat, " ", regex=True))

            stop = {"und","oder","der","die","das","ein","eine","eines","mit","im","in","auf","für","an","am",
                    "von","zu","zum","zur","ist","sind","war","waren",
                    "the","a","an","of","to","for","on","in","and","or","is","are"}
            word_re = re.compile(r"[A-Za-zÄÖÜäöüß]+")

            def tokens(txt): return [w for w in word_re.findall(txt.lower()) if w not in stop]
            all_tok = sum((tokens(t) for t in clean_resp), [])

            col1, col2 = st.columns(2)
            with col1:
                n_val = st.selectbox("📝 n-Gram length", [1,2,3,4,5], index=1)
            with col2:
                top_k = st.slider("🔢 Show top K n-grams", 5, 50, 20)

            def ngrams(lst,n): return zip(*[lst[i:] for i in range(n)])
            cnt = collections.Counter(" ".join(g) for g in ngrams(all_tok, n_val))
            ngram_df = (pd.DataFrame(cnt.items(), columns=["ngram","count"])
                        .sort_values("count", ascending=False).head(top_k))

            if ngram_df.empty:
                st.info("ℹ️ No n-grams found (try different settings).")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(ngram_df, use_container_width=True, hide_index=True)
                with col2:
                    st.altair_chart(
                        alt.Chart(ngram_df.head(10)).mark_bar()
                           .encode(y=alt.Y("ngram:N", sort="-x"), x="count:Q")
                           .properties(height=300),
                        use_container_width=True
                    )

                st.download_button(
                    f"📥 Download top {top_k} n-grams",
                    ngram_df.to_csv(index=False).encode("utf-8"),
                    f"{term}_top{top_k}_ngrams.csv", "text/csv",
                    key=f"dl_ngram_{term}_{n_val}_{top_k}"
                )

    # Export section
    st.markdown("---")
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        if not df_view.empty:
            st.dataframe(df_view.head(), use_container_width=True)
    with col2:
        st.download_button("📥 Download Filtered Data",
                           df_view.to_csv(index=False).encode("utf-8"),
                           "filtered_raw_data.csv", "text/csv",
                           key="dl_ngram_raw")

# ═════════════ TAB – Topic Clusters ═══════
with tab_topic:
    st.header("🎪 Topic Clusters (TF-IDF + K-Means)")

    if len(sel_terms) != 1:
        st.info("ℹ️ Please select exactly **one** Search Term in the sidebar.")
    else:
        term = sel_terms[0]
        df_term = df_view[df_view["Search Term"] == term]
        if df_term.empty:
            st.warning("⚠️ No responses for this term.")
        else:
            st.success(f"🎯 Analyzing topic clusters for: **{term}**")

            dom_pat = r"\b\w+\.(?:de|com|co\.uk|fr|es|it|nl|pl|cz|sk|hu|dk|fi|se|no|ch|be|bg|ro|pt|gr)\b"
            url_pat = r"http\S+"

            docs = (df_term["Response"].astype(str)
                    .str.replace(url_pat, " ", regex=True)
                    .str.replace(dom_pat, " ", regex=True))

            if len(docs) < 2:
                st.warning("⚠️ Need at least 2 responses for clustering.")
                st.info(f"Currently only {len(docs)} response(s) available for this search term.")
            else:
                max_k = min(10, len(docs))

                col1, col2 = st.columns(2)
                with col1:
                    k_val = st.slider("🎪 Number of clusters (K)", 2, max_k, min(4, max_k))
                with col2:
                    top_n = st.slider("🔍 Top terms per cluster", 3, 15, 7)
                tfidf = TfidfVectorizer(
                    stop_words=[
                        "und","oder","der","die","das","ein","eine","eines","mit","im","in","auf",
                        "für","an","am","von","zu","zum","zur","ist","sind","war","waren",
                        "the","a","an","of","to","for","on","in","and","or","is","are"
                    ],
                    ngram_range=(1,2),
                    min_df=2
                )

                try:
                    X = tfidf.fit_transform(docs)
                    km = KMeans(n_clusters=k_val, n_init="auto", random_state=0)
                    labels = km.fit_predict(X)

                    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
                    terms = tfidf.get_feature_names_out()

                    records=[]
                    for c in range(k_val):
                        top_words=", ".join(terms[i] for i in order_centroids[c, :top_n])
                        size=(labels==c).sum()
                        records.append({"Cluster":f"Cluster {c}","Size":size,"Top_Terms":top_words})

                    cluster_df = pd.DataFrame(records).sort_values("Size",ascending=False)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.dataframe(cluster_df, hide_index=True, use_container_width=True)
                    with col2:
                        st.altair_chart(
                            alt.Chart(cluster_df).mark_bar()
                               .encode(x=alt.X("Size:Q", title="Number of Responses"),
                                       y=alt.Y("Cluster:N", sort="-x"),
                                       tooltip=["Top_Terms"])
                               .properties(height=300),
                            use_container_width=True
                        )

                    st.markdown("#### 📝 Sample Response per Cluster")
                    for i, c in enumerate(cluster_df["Cluster"]):
                        cluster_num = int(c.split()[-1])
                        top_terms = cluster_df.iloc[i]["Top_Terms"]
                        example_idx = list(labels).index(cluster_num)

                        # Get the original row data including date
                        original_row = df_term.iloc[example_idx]
                        example_response = docs.iloc[example_idx]
                        response_date = original_row["Date"].strftime("%Y-%m-%d") if pd.notna(original_row["Date"]) else "No Date"
                        search_term = original_row["Search Term"]

                        with st.expander(f"**{c}**: *{top_terms}* | Date: {response_date}"):
                            st.markdown(f"**Search Term:** {search_term}")
                            st.markdown(f"**Date:** {response_date}")
                            st.markdown("**Full Response:**")
                            st.write(example_response)

                    # Enhanced download with cluster assignments
                    df_dl = df_term.copy()
                    df_dl["Cluster"] = [f"Cluster {label}" for label in labels]
                    df_dl["Cluster_Size"] = df_dl["Cluster"].map(cluster_df.set_index("Cluster")["Size"])

                    st.download_button(
                        "📥 Download Cluster Assignments",
                        df_dl.to_csv(index=False).encode("utf-8"),
                        f"{term}_clusters.csv",
                        "text/csv",
                        key="dl_topic_clusters"
                    )

                except Exception as e:
                    st.error(f"❌ Clustering failed: {str(e)}")
                    st.info("💡 Try reducing the number of clusters or check your data.")

    # Export section
    st.markdown("---")
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        if not df_view.empty:
            st.dataframe(df_view.head(), use_container_width=True)
    with col2:
        st.download_button("📥 Download Filtered Data",
                           df_view.to_csv(index=False).encode("utf-8"),
                           "filtered_raw_data.csv", "text/csv",
                           key="dl_topic_raw")

# ═════════════ Footer ═════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 <strong>AI Visibility Monitor</strong> | Enhanced Dashboard v2.0</p>
    <p>📊 Your domains are prioritized in all tables and charts | 🎯 Use filters to focus your analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)
