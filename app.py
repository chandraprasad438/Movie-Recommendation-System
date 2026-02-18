import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. INITIAL CONFIG & STATE =================
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# Initialize Session States
if 'page' not in st.session_state:
    st.session_state.page = "landing"
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

TMDB_API_KEY = "90af1a55371ef00d327b92d61b314d7b"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

# ================= 2. HELPERS =================
@st.cache_data(ttl=3600)
def fetch_api(url):
    try:
        res = requests.get(url).json()
        return res.get('results', [])
    except: return []

@st.cache_data(ttl=3600)
def get_full_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        d = requests.get(url).json()
        crew = d.get('credits', {}).get('crew', [])
        cast = d.get('credits', {}).get('cast', [])
        return {
            "dir": next((m['name'] for m in crew if m['job'] == 'Director'), "N/A"),
            "mus": next((m['name'] for m in crew if m['job'] in ['Music', 'Original Music Composer', 'Composer']), "N/A"),
            "star": ", ".join([m['name'] for m in cast[:2]]),
            "genre": ", ".join([g['name'] for g in d.get('genres', [])[:1]]),
            "year": d.get('release_date', '2026')[:4],
            "overview": d.get('overview', '')
        }
    except: return {"dir":"N/A","mus":"N/A","star":"N/A","genre":"N/A","year":"N/A", "overview":""}

# ================= 3. NLP ALGORITHM =================
def nlp_search(user_query, movies_list):
    """
    Simple Content-Based Filtering using TF-IDF and Cosine Similarity.
    It compares the user's query vector against movie overview vectors.
    """
    if not movies_list: return []
    
    # 1. Prepare Data
    descriptions = [m.get('overview', '') if m.get('overview') else "No description" for m in movies_list]
    
    # 2. Add User Query to the dataset temporarily
    # We append the user query at the end to vectorize it in the same space
    all_text = descriptions + [user_query]
    
    # 3. Vectorize (Convert Text to Numbers)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    # 4. Calculate Cosine Similarity
    # Compare the last item (user query) with all previous items (movies)
    # 
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # 5. Sort Scores
    scores = list(enumerate(cosine_sim[0]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # 6. Return Top 5 Matches
    top_indices = [i[0] for i in sorted_scores[:6] if i[1] > 0] # Only return if there is some similarity
    return [movies_list[i] for i in top_indices]

# ================= 4. ADVANCED NEON STYLE =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;500&display=swap');

    /* Global Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1478720568477-152d9b164e26?q=80&w=2000');
        background-size: cover; background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }

    /* Landing Page Specific - CENTERED */
    .landing-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 80vh; text-align: center;
    }
    .glitch-title {
        font-family: 'Orbitron', sans-serif; font-size: 80px; font-weight: 900;
        background: linear-gradient(90deg, #00dbde, #fc00ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 219, 222, 0.5);
        margin-bottom: 20px;
    }
    
    /* Neon Dividers & Headers */
    .neon-divider { height: 2px; background: linear-gradient(90deg, #00dbde, #fc00ff); margin: 20px 0 40px 0; box-shadow: 0 0 10px #00dbde; }
    .section-head { font-family: 'Orbitron', sans-serif; font-size: 24px; color: #fff; letter-spacing: 2px; text-transform: uppercase; }

    /* Movie Cards */
    .movie-card { background: rgba(20, 20, 20, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden; margin-bottom: 10px; transition: 0.3s; height: 100%; }
    .movie-card:hover { border-color: #00dbde; transform: translateY(-5px); }
    .movie-img { width: 100%; height: 220px; object-fit: cover; }
    .card-body { padding: 12px; }
    
    /* Title Styling */
    .m-title { 
        font-family: 'Orbitron', sans-serif; 
        font-size: 14px; 
        color: #fff; 
        white-space: normal !important; 
        overflow: visible !important;
        height: auto !important;
        min-height: 40px;
        line-height: 1.4;
        margin-bottom: 5px;
    }
    
    .m-sub { color: #00dbde; font-size: 11px; font-weight: bold; margin-bottom: 5px; }
    .m-info { font-size: 11px; color: #999; line-height: 1.5; }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #00dbde, #fc00ff) !important;
        color: white !important; border: none !important; font-family: 'Orbitron', sans-serif !important;
        border-radius: 5px !important; transition: 0.3s !important;
    }
    div.stButton > button:hover { opacity: 0.8; box-shadow: 0 0 15px #00dbde; }

    /* Sidebar Star Card */
    .sidebar-star-card { 
        background: #111; border: 1px solid #00dbde; 
        padding: 15px; border-radius: 8px; margin-top: 10px;
    }
    
    /* Input Fields */
    input[type="text"], textarea {
        background-color: #222 !important;
        color: #fff !important;
        border: 1px solid #444 !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= 5. LOGIC ROUTING =================

if st.session_state.page == "landing":
    # --- LANDING PAGE (PERFECTLY CENTERED) ---
    col_spacer_left, col_content, col_spacer_right = st.columns([1, 2, 1])
    
    with col_content:
        st.markdown("""
            <div style="height: 20vh;"></div>
            <div style="text-align: center;">
                <h1 class="glitch-title">Movie Recommendation System</h1>
                <h3 style="color:#fff; font-family:'Orbitron'; letter-spacing:3px;">UNLIMITED INDIAN ENTERTAINMENT</h3>
                <p style="color:#aaa; margin-top:20px; font-size:16px;">
                    Experience the magic of Cinema. Tamil, Telugu, Hindi & More. <br>
                    AI-Powered Recommendations | Future Tech Design
                </p>
            </div>
            <br><br>
        """, unsafe_allow_html=True)
        
        # Centered Button
        b1, b2, b3 = st.columns([1, 1, 1])
        with b2:
            if st.button("🚀 ENTER CINEMA"):
                st.session_state.page = "app"
                st.rerun()

else:
    # --- MAIN APP ---
    
    # SIDEBAR
    with st.sidebar:
        # 1. STAR PROFILE SECTION (FIXED & VISIBLE)
        st.markdown("<h2 style='font-family:Orbitron; color:#00dbde;'>⭐ STAR PROFILE</h2>", unsafe_allow_html=True)
        st.caption("Search for any Actor/Actress")
        star_q = st.text_input("Enter Name (e.g., Vijay)", "")
        
        if star_q:
            s_res = fetch_api(f"https://api.themoviedb.org/3/search/person?api_key={TMDB_API_KEY}&query={star_q}")
            if s_res:
                s_id = s_res[0]['id']
                sd = requests.get(f"https://api.themoviedb.org/3/person/{s_id}?api_key={TMDB_API_KEY}").json()
                st.markdown(f"""
                    <div class="sidebar-star-card">
                        <img src="{IMG_BASE + sd.get('profile_path') if sd.get('profile_path') else ''}" style="width:100%; border-radius:5px; margin-bottom:10px;">
                        <p style="font-family:Orbitron; color:#fff; font-size:18px; margin:0;">{sd['name']}</p>
                        <p style="color:#00dbde; font-size:12px; margin-bottom:5px;">{sd.get('known_for_department', 'Actor')}</p>
                        <hr style="border-color:#333;">
                        <p style="font-size:11px; color:#ccc;">
                            <b>Born:</b> {sd.get('birthday','N/A')}<br>
                            <b>Place:</b> {sd.get('place_of_birth','N/A')}
                        </p>
                        <p style="font-size:10px; color:#888; line-height:1.4;">
                            {sd.get('biography','No bio available.')[:200]}...
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Actor not found!")

        st.divider()

        # 2. WATCHLIST
        st.markdown("<h2 style='font-family:Orbitron; color:#00dbde;'>MY WATCHLIST</h2>", unsafe_allow_html=True)
        if st.session_state.watchlist:
            for idx, item in enumerate(st.session_state.watchlist):
                st.markdown(f"<div style='padding:8px 0; border-bottom:1px solid #222; font-size:13px;'>📍 {item}</div>", unsafe_allow_html=True)
            if st.button("🗑️ CLEAR LIST"):
                st.session_state.watchlist = []
                st.rerun()
        else: st.caption("No movies in your list.")
        
        st.divider()
        if st.button("🏠 BACK TO HOME"):
            st.session_state.page = "landing"
            st.rerun()

    # HERO SECTION
    st.markdown("""
    <div style="background: linear-gradient(90deg, #000 50%, transparent 100%), url('https://image.tmdb.org/t/p/original/zOpe0eHsq0A2HcNybArpazadQL3.jpg'); 
         background-size:cover; padding:80px 60px; border-radius:15px; border:1px solid #333; margin-bottom:20px;">
        <h1 style="font-family:Orbitron; font-size: 60px; font-weight: 900; margin:0; line-height:1;">MAHAVATAR</h1>
        <h2 style="font-family:Orbitron; font-size: 30px; color: #00dbde; margin:0; letter-spacing:10px;">NARASIMHA</h2>
        <p style="color: #aaa; max-width: 500px; margin-top:20px; font-size:16px;">The Legend Returns. Indian Epic Action.</p>
    </div>
    """, unsafe_allow_html=True)

    # REUSABLE ROW FUNCTION
    def render_row(movies, key):
        if not movies: 
            st.caption("No movies found or API limit reached.")
            return
        cols = st.columns(6)
        for i, m in enumerate(movies[:6]):
            with cols[i]:
                det = get_full_details(m['id'])
                poster = IMG_BASE + m['poster_path'] if m.get('poster_path') else "https://via.placeholder.com/500x750"
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster}" class="movie-img">
                        <div class="card-body">
                            <div class="m-sub">{det['year']} | {det['genre']}</div>
                            <div class="m-title">{m.get('title', m.get('name'))}</div>
                            <div class="m-info">
                                <b>Dir:</b> {det['dir'][:15]}<br>
                                <b style="color:#00dbde;">Mus:</b> {det['mus'][:15]}<br>
                                <b>Cast:</b> {det['star'][:18]}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1: st.link_button("▶", f"https://www.youtube.com/results?search_query={m.get('title','').replace(' ','+')}+trailer")
                with b2:
                    if st.button("➕", key=f"btn_{key}_{i}"):
                        if m.get('title') not in st.session_state.watchlist:
                            st.session_state.watchlist.append(m.get('title'))
                            st.rerun()

    def create_section(label, data, key):
        st.markdown(f"<div class='section-head'>{label}</div><div class='neon-divider'></div>", unsafe_allow_html=True)
        render_row(data, key)

    # 1. Trending (Strictly Indian)
    create_section("1. Trending Indian Hits", fetch_api(f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&region=IN&sort_by=popularity.desc&with_original_language=ta|te|hi|ml|kn"), "trend")

    # 2. Upcoming
    up_titles = ["Jana Nayagan", "Karuppu", "Jailer 2", "Toxic", "Ramayana", "Good Bad Ugly", "Viduthalai Part 2"]
    up_res = []
    for t in up_titles:
        r = fetch_api(f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={t}")
        if r: up_res.append(r[0])
    create_section("2. Upcoming Indian Movies", up_res, "upc")

    # 3. Mood
    st.markdown("<div class='section-head'>3. Emotion Detection Suggestion</div><div class='neon-divider'></div>", unsafe_allow_html=True)
    mood = st.selectbox("How is your vibe?", ["I need massive action movies!", "I am feeling sad / Need a cry", "I want to laugh out loud", "I am feeling romantic", "I need a mind-bending thriller"])
    m_id = {"I need massive action movies!": 28, "I am feeling sad / Need a cry": 18, "I want to laugh out loud": 35, "I am feeling romantic": 10749, "I need a mind-bending thriller": 53}[mood]
    render_row(fetch_api(f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={m_id}&region=IN&with_original_language=ta|te|hi"), "mood")

    # 4. Similar
    st.markdown("<div class='section-head'>4. Similar Movie Suggestions</div><div class='neon-divider'></div>", unsafe_allow_html=True)
    fav = st.text_input("Because you loved:", "Vikram")
    if fav:
        s_mv = fetch_api(f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={fav}")
        if s_mv: render_row(fetch_api(f"https://api.themoviedb.org/3/movie/{s_mv[0]['id']}/recommendations?api_key={TMDB_API_KEY}"), "sim")

    # 5. Anime
    create_section("5. Anime Movies", fetch_api(f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres=16"), "ani")

    # 6. Star Recent Hits
    st.markdown("<div class='section-head'>6. Indian Stars Recent Movies</div><div class='neon-divider'></div>", unsafe_allow_html=True)
    actor_inp = st.text_input("Enter Star Name for Recent Hits:", "Rajinikanth")
    if actor_inp:
        a_res = fetch_api(f"https://api.themoviedb.org/3/search/person?api_key={TMDB_API_KEY}&query={actor_inp}")
        if a_res: render_row(fetch_api(f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_cast={a_res[0]['id']}&sort_by=primary_release_date.desc"), "hits")

    # ================= 7. CINEMA ANALYTICS DASHBOARD =================
    st.markdown("<div class='section-head'>7. Cinema Data Analytics</div><div class='neon-divider'></div>", unsafe_allow_html=True)
    dash_data = fetch_api(f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&region=IN&sort_by=popularity.desc&with_original_language=ta|te|hi")
    
    if dash_data:
        df = pd.DataFrame([{"Title": m['title'], "Rating": m['vote_average'], "Year": m.get('release_date', '2025')[:4], "Pop": m['popularity'], "Votes": m['vote_count']} for m in dash_data])

        c1, c2, c3 = st.columns(3)
        with c1: 
            st.markdown("<h4 style='color:#00dbde'>1. Ratings Distribution</h4>", unsafe_allow_html=True)
            st.bar_chart(df['Rating'].value_counts(), color="#00dbde")
            
        with c2: 
            st.markdown("<h4 style='color:#fc00ff'>2. Top 10 Rated Movies</h4>", unsafe_allow_html=True)
            st.bar_chart(df.sort_values("Rating", ascending=False).head(10).set_index("Title")["Rating"], color="#fc00ff")
            
        with c3: 
            st.markdown("<h4 style='color:#fff'>3. Yearly Release Trend</h4>", unsafe_allow_html=True)
            st.line_chart(df['Year'].value_counts().sort_index(), color="#ffffff")
        
        # ROW 2 (3 Charts)
        c4, c5, c6 = st.columns(3)
        with c4: 
            st.markdown("<h4 style='color:#00dbde'>4. Popularity Volume</h4>", unsafe_allow_html=True)
            st.area_chart(df.set_index("Title")["Pop"].head(10), color="#00dbde")
            
        with c5: 
            st.markdown("<h4 style='color:#fc00ff'>5. Rating vs Popularity</h4>", unsafe_allow_html=True)
            st.scatter_chart(df, x="Rating", y="Pop", color="#fc00ff")
            
        with c6: 
            st.markdown("<h4 style='color:#ff4b4b'>6. Audience Engagement (Votes)</h4>", unsafe_allow_html=True)
            st.bar_chart(df.set_index("Title")["Votes"].head(7), color="#ff4b4b")
            
    # NLP SECTION (FIXED & IMPROVED)
    st.markdown("<div class='section-head'>8. AI NLP Semantic Search 🤖</div><div class='neon-divider'></div>", unsafe_allow_html=True)
    st.info("💡 **AI Search Engine:** Describe a plot like 'Space scientist' or 'Ghost horror' to find matches from 200+ movies.")
    
    nlp_query = st.text_area("What kind of movie plot are you looking for?", "A policeman fighting a gangster for revenge in a big city", height=100)
    
    if st.button("🔍 ANALYZE WITH NLP"):
        with st.spinner("AI is reading 200+ movie scripts..."):
            # Fetch 10 pages (200 movies) to cover all genres like Sci-Fi and Horror
            all_movies = []
            for p in range(1, 11):
                url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&region=IN&sort_by=popularity.desc&page={p}&with_original_language=ta|te|hi|en"
                all_movies.extend(fetch_api(url))
            
            # Remove duplicates
            unique_pool = list({m['id']: m for m in all_movies}.values())
            
            results = nlp_search(nlp_query, unique_pool)
            
            if results:
                st.success(f"Analysis Complete! Found {len(results)} matches.")
                render_row(results, "nlp")
            else:
                st.error("No exact matches in the current trending list. Try different keywords like 'Time travel', 'Ghost', or 'Revenge'.")


