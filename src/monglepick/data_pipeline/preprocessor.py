"""
데이터 전처리기.

§11-6 전처리기 상세:
1. 장르 한국어 변환 (TMDB genre ID → 한국어)
2. 무드태그 생성 (Ollama qwen2.5:14b, 25개 화이트리스트)
3. 임베딩 입력 텍스트 구성 (구조화된 텍스트)
4. OTT 플랫폼명 정규화 ("Netflix" → "넷플릭스")

§11-3 데이터 검증 규칙:
- 필수 필드: id, title, genres, release_year
- 평점 범위: 0~10
- 개봉년도: 1900~현재
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from monglepick.config import settings
from monglepick.data_pipeline.models import MovieDocument, TMDBRawMovie

logger = structlog.get_logger()

# ============================================================
# 상수: 장르 매핑 테이블
# ============================================================

# TMDB genre_id → 한국어 (§11-6 단계 1)
GENRE_ID_TO_KR: dict[int, str] = {
    28: "액션", 12: "모험", 16: "애니메이션", 35: "코미디",
    80: "범죄", 99: "다큐멘터리", 18: "드라마", 10751: "가족",
    14: "판타지", 36: "역사", 27: "공포", 10402: "음악",
    9648: "미스터리", 10749: "로맨스", 878: "SF", 10770: "TV 영화",
    53: "스릴러", 10752: "전쟁", 37: "서부",
}

# 영문 장르명 → 한국어 (KOBIS/Kaggle 보강용)
GENRE_EN_TO_KR: dict[str, str] = {
    "Action": "액션", "Adventure": "모험", "Animation": "애니메이션",
    "Comedy": "코미디", "Crime": "범죄", "Documentary": "다큐멘터리",
    "Drama": "드라마", "Family": "가족", "Fantasy": "판타지",
    "History": "역사", "Horror": "공포", "Music": "음악",
    "Mystery": "미스터리", "Romance": "로맨스", "Science Fiction": "SF",
    "TV Movie": "TV 영화", "Thriller": "스릴러", "War": "전쟁", "Western": "서부",
}

# ============================================================
# 상수: 무드태그 관련
# ============================================================

# 25개 허용 무드태그 화이트리스트 (§11-6)
MOOD_WHITELIST: set[str] = {
    "몰입", "감동", "웅장", "긴장감", "힐링", "유쾌", "따뜻", "슬픔",
    "공포", "잔잔", "스릴", "카타르시스", "청춘", "우정", "가족애",
    "로맨틱", "미스터리", "반전", "철학적", "사회비판", "모험", "판타지",
    "레트로", "다크", "유머",
}

# §11-6-1 장르 → 무드 기본 매핑 테이블 (GPT 실패 시 fallback)
GENRE_TO_DEFAULT_MOOD: dict[str, list[str]] = {
    "액션": ["몰입", "스릴"], "모험": ["모험", "몰입"], "애니메이션": ["따뜻", "판타지"],
    "코미디": ["유쾌", "유머"], "범죄": ["긴장감", "다크"], "다큐멘터리": ["철학적", "사회비판"],
    "드라마": ["감동", "잔잔"], "가족": ["가족애", "따뜻"], "판타지": ["판타지", "모험"],
    "역사": ["웅장", "감동"], "공포": ["공포", "다크"], "음악": ["감동", "힐링"],
    "미스터리": ["미스터리", "긴장감"], "로맨스": ["로맨틱", "따뜻"], "SF": ["몰입", "웅장"],
    "TV 영화": ["잔잔"], "스릴러": ["스릴", "긴장감"], "전쟁": ["웅장", "카타르시스"],
    "서부": ["모험", "레트로"],
}

# ============================================================
# 상수: OTT 플랫폼 정규화 테이블
# ============================================================

# §11-6 단계 4: 영문 → 한국어
OTT_NORMALIZE: dict[str, str] = {
    "Netflix": "넷플릭스", "Disney Plus": "디즈니+", "Amazon Prime Video": "아마존 프라임",
    "Wavve": "웨이브", "Watcha": "왓챠", "Tving": "티빙",
    "Coupang Play": "쿠팡플레이", "Apple TV Plus": "애플TV+", "Apple TV": "애플TV+",
    "Google Play Movies": "구글플레이", "YouTube": "유튜브",
    "Naver Store": "네이버", "KakaoPage": "카카오페이지",
}

# ============================================================
# 상수: TMDB 키워드 영→한 매핑 테이블 (Phase ML-2)
# ============================================================
# TMDB API /movie/{id}/keywords는 항상 영문 키워드를 반환한다.
# 사용자가 한국어로 검색할 때 BM25/벡터 매칭이 가능하도록
# 주요 키워드 ~200개를 한국어로 매핑한다.
# 매핑에 없는 키워드는 영문 원본을 유지한다 (벡터 검색의 cross-lingual 특성 활용).

KEYWORD_EN_TO_KR: dict[str, str] = {
    # ── 테마/분위기 ──
    "based on novel": "소설 원작", "based on novel or book": "소설 원작",
    "based on true story": "실화", "based on a true story": "실화",
    "based on comic": "만화 원작", "based on comic book": "만화 원작",
    "based on manga": "만화 원작",
    "revenge": "복수", "redemption": "구원", "survival": "생존",
    "dystopia": "디스토피아", "utopia": "유토피아",
    "coming of age": "성장", "coming-of-age": "성장",
    "underdog": "약자의 반전", "friendship": "우정",
    "love": "사랑", "love triangle": "삼각관계",
    "forbidden love": "금지된 사랑", "unrequited love": "짝사랑",
    "family": "가족", "family relationships": "가족 관계",
    "father son relationship": "부자 관계", "mother daughter relationship": "모녀 관계",
    "sibling relationship": "형제자매",
    "loneliness": "외로움", "isolation": "고립",
    "grief": "슬픔", "loss": "상실", "death": "죽음",
    "betrayal": "배신", "jealousy": "질투",
    "corruption": "부패", "conspiracy": "음모",
    "sacrifice": "희생", "faith": "신앙",
    "ambition": "야망", "obsession": "집착",
    "identity": "정체성", "self-discovery": "자아 발견",
    "fate": "운명", "destiny": "운명",
    "dream": "꿈", "nightmare": "악몽",
    "memory": "기억", "amnesia": "기억상실",
    "time": "시간", "time travel": "시간 여행", "time loop": "타임루프",
    "parallel universe": "평행 우주", "alternate reality": "대체 현실",
    "afterlife": "사후세계",
    # ── 장소/배경 ──
    "new york": "뉴욕", "new york city": "뉴욕", "los angeles": "로스앤젤레스",
    "london": "런던", "paris": "파리", "tokyo": "도쿄", "seoul": "서울",
    "high school": "고등학교", "college": "대학", "university": "대학",
    "prison": "감옥", "hospital": "병원",
    "island": "섬", "desert": "사막", "jungle": "정글", "forest": "숲",
    "ocean": "바다", "mountain": "산", "small town": "시골 마을",
    "space": "우주", "outer space": "우주", "spaceship": "우주선",
    "mars": "화성", "moon": "달",
    "castle": "성", "palace": "궁전",
    "submarine": "잠수함", "airplane": "비행기", "train": "기차",
    "haunted house": "귀신 집",
    # ── 시대/역사 ──
    "world war ii": "제2차 세계대전", "world war i": "제1차 세계대전",
    "cold war": "냉전", "vietnam war": "베트남 전쟁",
    "korean war": "한국전쟁", "civil war": "내전",
    "medieval": "중세", "middle ages": "중세",
    "ancient rome": "고대 로마", "ancient egypt": "고대 이집트",
    "roman empire": "로마 제국",
    "1920s": "1920년대", "1930s": "1930년대", "1940s": "1940년대",
    "1950s": "1950년대", "1960s": "1960년대", "1970s": "1970년대",
    "1980s": "1980년대", "1990s": "1990년대",
    "historical fiction": "역사 소설", "period drama": "시대극",
    # ── 직업/신분 ──
    "detective": "탐정", "police": "경찰", "cop": "경찰",
    "spy": "스파이", "secret agent": "비밀 요원",
    "soldier": "군인", "warrior": "전사",
    "assassin": "암살자", "hitman": "킬러",
    "serial killer": "연쇄 살인범", "murder": "살인",
    "thief": "도둑", "heist": "강도", "bank robbery": "은행 강도",
    "lawyer": "변호사", "judge": "판사",
    "doctor": "의사", "nurse": "간호사",
    "teacher": "교사", "student": "학생",
    "journalist": "기자", "writer": "작가",
    "musician": "음악가", "singer": "가수", "dancer": "댄서",
    "athlete": "운동선수", "boxer": "권투 선수",
    "king": "왕", "queen": "여왕", "princess": "공주", "prince": "왕자",
    "vampire": "뱀파이어", "zombie": "좀비", "werewolf": "늑대인간",
    "ghost": "유령", "witch": "마녀", "wizard": "마법사",
    "robot": "로봇", "android": "안드로이드", "cyborg": "사이보그",
    "alien": "외계인", "extraterrestrial": "외계 생명체",
    "superhero": "슈퍼히어로", "supervillain": "슈퍼빌런",
    "pirate": "해적", "cowboy": "카우보이", "samurai": "사무라이",
    "ninja": "닌자",
    # ── 소재/장치 ──
    "artificial intelligence": "인공지능", "virtual reality": "가상현실",
    "hacking": "해킹", "computer": "컴퓨터", "technology": "기술",
    "nuclear": "핵", "nuclear war": "핵전쟁",
    "pandemic": "팬데믹", "virus": "바이러스", "plague": "전염병",
    "magic": "마법", "sorcery": "마법", "supernatural": "초자연",
    "prophecy": "예언", "curse": "저주",
    "treasure": "보물", "treasure hunt": "보물 찾기",
    "sword": "검", "martial arts": "무술", "kung fu": "쿵후",
    "car chase": "자동차 추격", "race": "레이스",
    "drug": "마약", "drug dealing": "마약 거래",
    "gambling": "도박", "casino": "카지노",
    "music": "음악", "rock music": "록 음악", "jazz": "재즈",
    "dance": "춤", "ballet": "발레",
    "sport": "스포츠", "football": "축구", "baseball": "야구",
    "basketball": "농구", "boxing": "권투",
    "cooking": "요리", "food": "음식",
    "fashion": "패션", "photography": "사진",
    "painting": "그림", "art": "예술",
    "wedding": "결혼", "divorce": "이혼", "pregnancy": "임신",
    "adoption": "입양",
    # ── 분위기/스타일 ──
    "dark comedy": "블랙 코미디", "satire": "풍자", "parody": "패러디",
    "slapstick": "슬랩스틱", "romantic comedy": "로맨틱 코미디",
    "psychological thriller": "심리 스릴러", "neo-noir": "네오 느와르",
    "film noir": "필름 느와르", "noir": "느와르",
    "mockumentary": "모큐멘터리",
    "found footage": "파운드 푸티지",
    "anthology": "옴니버스", "nonlinear timeline": "비선형 서사",
    "plot twist": "반전", "surprise ending": "반전 결말",
    "cliffhanger": "클리프행어",
    "slow burn": "슬로우 번", "suspense": "서스펜스",
    "gore": "고어", "torture": "고문", "violence": "폭력",
    "erotic": "에로틱", "sexuality": "성적",
    "animation": "애니메이션", "stop motion": "스톱모션",
    "musical": "뮤지컬", "concert": "콘서트",
    # ── 한국 특화 테마 ──
    "korean": "한국", "south korea": "한국",
    "k-drama": "한국 드라마", "joseon dynasty": "조선시대",
    "gangster": "조폭", "organized crime": "조직 범죄",
    "chaebol": "재벌", "military service": "군대",
    # ── 일본/아시아 특화 ──
    "anime": "애니메이션", "manga": "만화",
    "yakuza": "야쿠자", "triad": "삼합회",
    "wuxia": "무협", "xianxia": "선협",
    # ── 미국 하이틴/청소년 ──
    "prom": "프롬", "teenager": "10대", "teen": "10대",
    "cheerleader": "치어리더", "fraternity": "대학 동아리",
    "bullying": "괴롭힘", "popularity": "인기",
    "first love": "첫사랑", "summer vacation": "여름 방학",
    "road trip": "로드 트립", "camping": "캠핑",
}

# 소문자 키 기반 빠른 조회용 (대소문자 무시 매칭)
_KEYWORD_EN_TO_KR_LOWER: dict[str, str] = {k.lower(): v for k, v in KEYWORD_EN_TO_KR.items()}


# ============================================================
# 전처리 함수
# ============================================================


def convert_genres(raw_genres: list[dict]) -> list[str]:
    """
    TMDB 장르 객체 배열을 한국어 장르 문자열 리스트로 변환한다.

    §11-6 단계 1: TMDB genre ID → 한국어.
    변환 실패 시 영문명 유지 (§11-3).

    Args:
        raw_genres: [{"id": 878, "name": "Science Fiction"}, ...]

    Returns:
        ["SF", "드라마", ...]
    """
    result: list[str] = []
    for genre in raw_genres:
        genre_id = genre.get("id")
        genre_name = genre.get("name", "")

        # 1차: ID로 변환
        if genre_id and genre_id in GENRE_ID_TO_KR:
            result.append(GENRE_ID_TO_KR[genre_id])
        # 2차: 영문명으로 변환
        elif genre_name in GENRE_EN_TO_KR:
            result.append(GENRE_EN_TO_KR[genre_name])
        # 3차: 변환 실패 시 영문명 유지
        elif genre_name:
            result.append(genre_name)

    return result


def extract_director(credits: dict) -> str:
    """credits.crew에서 job='Director'인 사람의 이름을 추출한다."""
    crew = credits.get("crew", [])
    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "")
    return ""


def extract_director_names(credits: dict) -> tuple[str, str]:
    """
    credits.crew에서 감독의 name과 original_name을 모두 추출한다.

    Phase ML-2: TMDB API는 ko-KR 요청 시 name에 한국어 이름(번역 있으면)을,
    original_name에 원어 이름을 반환한다.
    예) name="크리스토퍼 놀란", original_name="Christopher Nolan"
    번역이 없으면 name=original_name="Christopher Nolan" (동일).

    Returns:
        (name, original_name) 튜플. 같으면 original_name은 빈 문자열.
    """
    crew = credits.get("crew", [])
    for person in crew:
        if person.get("job") == "Director":
            name = person.get("name", "")
            original_name = person.get("original_name", "")
            # name과 original_name이 같으면 중복 저장 방지
            if original_name == name:
                original_name = ""
            return name, original_name
    return "", ""


def extract_cast(credits: dict, top_n: int = 5) -> list[str]:
    """credits.cast에서 상위 N명 배우 이름을 추출한다."""
    cast = credits.get("cast", [])
    return [person.get("name", "") for person in cast[:top_n] if person.get("name")]


def extract_cast_names(credits: dict, top_n: int = 5) -> list[str]:
    """
    credits.cast에서 상위 N명의 name + original_name을 모두 추출한다.

    Phase ML-2: 한글 이름과 영문 이름을 모두 포함하여
    "톰 크루즈" 또는 "Tom Cruise" 어느 쪽으로 검색해도 매칭되도록 한다.

    Returns:
        중복 제거된 이름 리스트 (최대 top_n*2개)
        예: ["톰 크루즈", "Tom Cruise", "마일스 텔러", "Miles Teller", ...]
    """
    cast = credits.get("cast", [])
    result: list[str] = []
    seen: set[str] = set()

    for person in cast[:top_n]:
        name = person.get("name", "")
        original_name = person.get("original_name", "")

        if name and name not in seen:
            result.append(name)
            seen.add(name)
        # original_name이 name과 다르면 추가 (한/영 이중 저장)
        if original_name and original_name != name and original_name not in seen:
            result.append(original_name)
            seen.add(original_name)

    return result


def extract_keywords(keywords_data: dict) -> list[str]:
    """
    keywords 객체에서 키워드 이름 리스트를 추출한다.

    Phase ML-2: TMDB API는 항상 영문 키워드를 반환하므로,
    KEYWORD_EN_TO_KR 매핑 테이블로 한국어 변환을 시도한다.
    매핑에 없는 키워드는 영문 원본과 함께 반환하여
    벡터 검색의 cross-lingual 특성을 활용한다.

    Returns:
        한국어 변환된 키워드 + 원본 영문 키워드 (중복 제거)
        예: ["시간 여행", "time travel", "디스토피아", "dystopia", "rebellion"]
    """
    keywords = keywords_data.get("keywords", [])
    result: list[str] = []
    seen: set[str] = set()

    for kw in keywords:
        name = kw.get("name", "")
        if not name:
            continue

        # 한국어 매핑이 있으면 한국어를 우선 추가
        kr_name = _KEYWORD_EN_TO_KR_LOWER.get(name.lower())
        if kr_name and kr_name not in seen:
            result.append(kr_name)
            seen.add(kr_name)

        # 영문 원본도 추가 (벡터 검색 + 영문 쿼리 대응)
        if name not in seen:
            result.append(name)
            seen.add(name)

    return result


# ============================================================
# Phase B: 확장 크레딧 추출 함수
# ============================================================


def extract_cast_with_characters(credits: dict, top_n: int = 5) -> list[dict]:
    """
    credits.cast에서 상위 N명 배우의 상세 정보를 추출한다.

    Phase C 확장: id, profile_path, gender, popularity, original_name, order 추가.
    모든 필드를 dict에 포함하여 Person 노드 품질 개선 (ID 기반 중복 방지, 사진 표시 등).
    """
    cast = credits.get("cast", [])
    return [
        {
            "id": p.get("id", 0),  # Phase C: TMDB person ID
            "name": p.get("name", ""),
            "character": p.get("character", ""),
            "profile_path": p.get("profile_path") or "",  # Phase C: 프로필 사진
            "gender": p.get("gender", 0),  # Phase C: 0=미지정, 1=여성, 2=남성
            "popularity": p.get("popularity", 0.0),  # Phase C: 인기도
            "original_name": p.get("original_name", ""),  # Phase C: 원어 이름
            "order": p.get("order", 0),  # Phase C: 빌링 순서
        }
        for p in cast[:top_n]
        if p.get("name")
    ]


def extract_cinematographer(credits: dict) -> str:
    """credits.crew에서 촬영감독(Director of Photography)을 추출한다."""
    for person in credits.get("crew", []):
        if person.get("job") == "Director of Photography":
            return person.get("name", "")
    return ""


def extract_composer(credits: dict) -> str:
    """credits.crew에서 음악감독/작곡가를 추출한다."""
    for person in credits.get("crew", []):
        if person.get("job") in ("Original Music Composer", "Music"):
            return person.get("name", "")
    return ""


def extract_screenwriters(credits: dict) -> list[str]:
    """credits.crew에서 각본가 목록을 추출한다."""
    writers: list[str] = []
    for person in credits.get("crew", []):
        if person.get("job") in ("Screenplay", "Writer") and person.get("name"):
            if person["name"] not in writers:
                writers.append(person["name"])
    return writers


def extract_producers(credits: dict) -> list[str]:
    """credits.crew에서 프로듀서 목록을 추출한다 (Producer 직급만)."""
    return [
        p.get("name", "")
        for p in credits.get("crew", [])
        if p.get("job") == "Producer" and p.get("name")
    ]


def extract_editor(credits: dict) -> str:
    """credits.crew에서 편집자를 추출한다."""
    for person in credits.get("crew", []):
        if person.get("job") == "Editor":
            return person.get("name", "")
    return ""


# ============================================================
# Phase C: 감독 확장 정보 + 추가 크루 추출
# ============================================================


def extract_director_details(credits: dict) -> dict:
    """
    credits.crew에서 감독의 상세 정보를 추출한다.

    Phase C: TMDB person ID, 프로필 사진, 원어 이름 포함.
    Person 노드에서 이름 충돌 방지를 위해 ID를 활용한다.

    Returns:
        {"id": int, "name": str, "profile_path": str, "original_name": str}
    """
    for person in credits.get("crew", []):
        if person.get("job") == "Director":
            return {
                "id": person.get("id", 0),
                "name": person.get("name", ""),
                "profile_path": person.get("profile_path") or "",
                "original_name": person.get("original_name", ""),
            }
    return {"id": 0, "name": "", "profile_path": "", "original_name": ""}


def extract_executive_producers(credits: dict) -> list[str]:
    """credits.crew에서 총괄 프로듀서 목록을 추출한다."""
    return [
        p.get("name", "")
        for p in credits.get("crew", [])
        if p.get("job") == "Executive Producer" and p.get("name")
    ]


def extract_production_designer(credits: dict) -> str:
    """credits.crew에서 프로덕션 디자이너를 추출한다."""
    for person in credits.get("crew", []):
        if person.get("job") == "Production Design":
            return person.get("name", "")
    return ""


def extract_costume_designer(credits: dict) -> str:
    """credits.crew에서 의상 디자이너를 추출한다."""
    for person in credits.get("crew", []):
        if person.get("job") in ("Costume Design", "Costume Designer"):
            return person.get("name", "")
    return ""


def extract_source_author(credits: dict) -> str:
    """credits.crew에서 원작 작가를 추출한다 (Novel/Characters 직군)."""
    for person in credits.get("crew", []):
        if person.get("job") in ("Novel", "Characters", "Original Story", "Story"):
            return person.get("name", "")
    return ""


# ============================================================
# Phase C: 이미지 / 대체 제목 / 추천 / KR 개봉일 추출
# ============================================================


def extract_images(raw_images: dict) -> tuple[list[str], list[str]]:
    """
    TMDB images에서 포스터/배경 이미지 경로를 추출한다 (각 최대 10개).

    Returns:
        (posters, backdrops) 튜플
    """
    posters = raw_images.get("posters", [])
    backdrops = raw_images.get("backdrops", [])
    return (
        [img for img in posters[:10] if img],
        [img for img in backdrops[:10] if img],
    )


def extract_kr_release_date(raw_release_dates: list[dict]) -> str:
    """
    release_dates에서 한국(KR) 극장 개봉일을 추출한다.

    release type 3 (Theatrical) 우선, 없으면 type 4 (Digital), 없으면 첫 번째.
    KR 데이터가 없으면 빈 문자열 반환.

    Returns:
        "YYYY-MM-DD" 형식 개봉일 또는 빈 문자열
    """
    for entry in raw_release_dates:
        if entry.get("iso_3166_1") == "KR":
            releases = entry.get("release_dates", [])
            # type 3 (Theatrical) 우선
            for r in releases:
                if r.get("type") == 3:
                    date_str = r.get("release_date", "")
                    if date_str:
                        return date_str[:10]  # ISO 날짜에서 YYYY-MM-DD만 추출
            # type 4 (Digital) fallback
            for r in releases:
                if r.get("type") == 4:
                    date_str = r.get("release_date", "")
                    if date_str:
                        return date_str[:10]
            # 아무 타입이나 첫 번째
            if releases:
                date_str = releases[0].get("release_date", "")
                if date_str:
                    return date_str[:10]
    return ""


def extract_collection_images(raw_collection: dict | None) -> tuple[str, str]:
    """
    TMDB belongs_to_collection에서 컬렉션 포스터/배경 이미지를 추출한다.

    Returns:
        (poster_path, backdrop_path) 튜플
    """
    if not raw_collection:
        return "", ""
    return (
        raw_collection.get("poster_path") or "",
        raw_collection.get("backdrop_path") or "",
    )


def extract_production_companies_full(raw_companies: list[dict]) -> list[dict]:
    """
    TMDB production_companies를 {id, name, logo_path, origin_country} 딕셔너리 리스트로 정규화한다.

    Phase C: logo_path와 origin_country 추가.
    """
    return [
        {
            "id": c.get("id", 0),
            "name": c.get("name", ""),
            "logo_path": c.get("logo_path") or "",  # Phase C: 제작사 로고
            "origin_country": c.get("origin_country", ""),  # Phase C: 제작사 본국
        }
        for c in raw_companies
        if c.get("name")
    ]


def extract_production_country_names(raw_countries: list[dict]) -> list[str]:
    """TMDB production_countries에서 국가 전체 이름 리스트를 추출한다."""
    return [
        c.get("name", "")
        for c in raw_countries
        if c.get("name")
    ]


def extract_spoken_language_names(raw_languages: list[dict]) -> list[str]:
    """TMDB spoken_languages에서 언어 전체 이름 리스트를 추출한다."""
    names: list[str] = []
    for lang in raw_languages:
        # english_name 우선, 없으면 name (native name) 사용
        name = lang.get("english_name") or lang.get("name", "")
        if name:
            names.append(name)
    return names


# ============================================================
# Phase D: translations / external_ids / images_logos 추출
# ============================================================


def extract_overview_from_translations(
    translations: list[dict],
    current_overview: str,
) -> tuple[str, str, str]:
    """
    translations에서 영문/일본어 줄거리를 추출하고, overview가 빈 경우 보강한다.

    Phase D: overview 빈 영화에 대해 영어/일본어 번역에서 보강.

    Args:
        translations: TMDBRawMovie.translations (전체 dict 리스트)
        current_overview: 현재 한국어 overview (비어있을 수 있음)

    Returns:
        (보강된 overview, overview_en, overview_ja) 튜플
    """
    overview_en = ""
    overview_ja = ""

    for t in translations:
        lang = t.get("iso_639_1", "")
        data = t.get("data", {})
        overview_text = data.get("overview", "") if isinstance(data, dict) else ""

        if lang == "en" and overview_text and not overview_en:
            overview_en = overview_text
        elif lang == "ja" and overview_text and not overview_ja:
            overview_ja = overview_text

    # overview가 비어있으면 영어 → 일본어 순으로 fallback
    boosted_overview = current_overview
    if not boosted_overview or len(boosted_overview) < 10:
        if overview_en:
            boosted_overview = overview_en
        elif overview_ja:
            boosted_overview = overview_ja

    return boosted_overview, overview_en, overview_ja


def extract_external_ids_full(external_ids: dict) -> dict:
    """
    external_ids raw dict에서 소셜 미디어 / 외부 DB ID를 추출한다.

    Phase D: 기존 5개만 → raw dict 전체에서 필요한 ID를 추출.

    Args:
        external_ids: TMDBRawMovie.external_ids (raw dict 전체)

    Returns:
        {"facebook_id": "...", "instagram_id": "...", "twitter_id": "...", "wikidata_id": "..."}
    """
    return {
        "facebook_id": external_ids.get("facebook_id") or "",
        "instagram_id": external_ids.get("instagram_id") or "",
        "twitter_id": external_ids.get("twitter_id") or "",
        "wikidata_id": external_ids.get("wikidata_id") or "",
    }


def extract_images_logos(raw_images: dict) -> list[str]:
    """
    TMDB images에서 로고 이미지 경로를 추출한다.

    Args:
        raw_images: TMDBRawMovie.images dict

    Returns:
        로고 이미지 경로 리스트
    """
    return raw_images.get("logos", [])


def extract_tmdb_list_count(lists_data: dict) -> int:
    """
    TMDB lists에서 이 영화가 포함된 리스트 총 수를 추출한다.

    Args:
        lists_data: TMDBRawMovie.lists dict

    Returns:
        총 리스트 수 (total_results)
    """
    return lists_data.get("total_results", 0)


# ============================================================
# Phase B: 컬렉션/제작사/국가/언어 추출 함수
# ============================================================


def extract_collection(raw_collection: dict | None) -> tuple[int, str]:
    """TMDB belongs_to_collection에서 컬렉션 ID와 이름을 추출한다."""
    if not raw_collection:
        return 0, ""
    return (
        raw_collection.get("id", 0) or 0,
        raw_collection.get("name", "") or "",
    )


def extract_production_companies(raw_companies: list[dict]) -> list[dict]:
    """TMDB production_companies를 {id, name} 딕셔너리 리스트로 정규화한다."""
    return [
        {"id": c.get("id", 0), "name": c.get("name", "")}
        for c in raw_companies
        if c.get("name")
    ]


def extract_production_countries(raw_countries: list[dict]) -> list[str]:
    """TMDB production_countries에서 ISO 코드 리스트를 추출한다."""
    return [
        c.get("iso_3166_1", "")
        for c in raw_countries
        if c.get("iso_3166_1")
    ]


def extract_spoken_languages(raw_languages: list[dict]) -> list[str]:
    """TMDB spoken_languages에서 ISO 코드 리스트를 추출한다."""
    return [
        lang.get("iso_639_1", "")
        for lang in raw_languages
        if lang.get("iso_639_1")
    ]


def normalize_ott_platforms(watch_providers: dict) -> list[str]:
    """
    TMDB watch/providers 응답에서 한국(KR) OTT 플랫폼 목록을 추출하고 한국어로 정규화한다.

    §11-6 단계 4: "Netflix" → "넷플릭스"
    """
    kr_providers = watch_providers.get("KR", {})
    flatrate = kr_providers.get("flatrate", [])

    result: list[str] = []
    for provider in flatrate:
        name = provider.get("provider_name", "")
        # 정규화 매핑이 있으면 적용, 없으면 원본 유지
        normalized = OTT_NORMALIZE.get(name, name)
        if normalized and normalized not in result:
            result.append(normalized)

    return result


# ============================================================
# Phase A: TMDB 보강 데이터 추출 함수
# ============================================================


def extract_reviews(raw_reviews: list[dict], max_count: int = 5, max_len: int = 500) -> list[str]:
    """
    리뷰 텍스트를 추출하고 길이를 제한한다.

    rating이 높은 순으로 정렬하여 상위 max_count개를 반환한다.
    rating이 None인 리뷰는 rating=0으로 간주하여 뒤로 밀린다.

    Args:
        raw_reviews: [{"author": "...", "content": "...", "rating": 8.0}, ...]
        max_count: 최대 리뷰 수 (기본 5)
        max_len: 리뷰당 최대 글자 수 (기본 500)

    Returns:
        리뷰 텍스트 리스트 (길이 제한 적용)
    """
    # rating 높은 순 정렬 (None → 0으로 처리)
    sorted_reviews = sorted(
        raw_reviews,
        key=lambda r: (r.get("author_details") or {}).get("rating") or r.get("rating") or 0,
        reverse=True,
    )

    result: list[str] = []
    for review in sorted_reviews[:max_count]:
        content = review.get("content", "").strip()
        if content:
            # 최대 길이 제한
            result.append(content[:max_len])

    return result


def extract_trailer_url(raw_videos: list[dict]) -> str:
    """
    YouTube 트레일러 URL을 추출한다.

    우선순위: Trailer > Teaser. YouTube가 아닌 사이트는 무시.
    동일 타입이 여러 개면 첫 번째를 사용한다.

    Args:
        raw_videos: [{"key": "dQw4...", "type": "Trailer", "site": "YouTube"}, ...]

    Returns:
        YouTube URL (예: "https://www.youtube.com/watch?v=dQw4...") 또는 빈 문자열
    """
    # YouTube 비디오만 필터링
    youtube_videos = [v for v in raw_videos if v.get("site") == "YouTube" and v.get("key")]

    # Trailer 우선 탐색
    for video in youtube_videos:
        if video.get("type") == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"

    # Trailer 없으면 Teaser
    for video in youtube_videos:
        if video.get("type") == "Teaser":
            return f"https://www.youtube.com/watch?v={video['key']}"

    return ""


def extract_behind_the_scenes(raw_videos: list[dict]) -> list[str]:
    """
    비하인드/피처렛/메이킹 YouTube URL을 추출한다.

    대상 타입: Behind the Scenes, Featurette

    Args:
        raw_videos: [{"key": "...", "type": "Behind the Scenes", "site": "YouTube"}, ...]

    Returns:
        YouTube URL 리스트
    """
    bts_types = {"Behind the Scenes", "Featurette"}
    result: list[str] = []

    for video in raw_videos:
        if (
            video.get("site") == "YouTube"
            and video.get("key")
            and video.get("type") in bts_types
        ):
            result.append(f"https://www.youtube.com/watch?v={video['key']}")

    return result


def extract_certification(raw_release_dates: list[dict], country: str = "KR") -> str:
    """
    한국(KR) 관람등급을 추출한다. KR 데이터가 없으면 US fallback.

    TMDB release_dates.results 형식:
    [{"iso_3166_1": "KR", "release_dates": [{"certification": "15", "type": 3}]}, ...]

    TMDB certification 값을 한국어로 매핑:
    - "All" / "" → "전체 관람가"
    - "12" → "12세 이상 관람가"
    - "15" → "15세 이상 관람가"
    - "18" / "R" → "청소년 관람불가"
    - "Restricted Screening" → "제한상영가"

    Args:
        raw_release_dates: TMDB release_dates.results 배열
        country: 우선 탐색 국가 (기본 "KR")

    Returns:
        한국어 관람등급 문자열 또는 빈 문자열
    """
    # KR 관람등급 → 한국어 매핑
    cert_kr_map: dict[str, str] = {
        "All": "전체 관람가",
        "전체 관람가": "전체 관람가",
        "12": "12세 이상 관람가",
        "12세 이상 관람가": "12세 이상 관람가",
        "15": "15세 이상 관람가",
        "15세 이상 관람가": "15세 이상 관람가",
        "18": "청소년 관람불가",
        "청소년 관람불가": "청소년 관람불가",
        "Restricted Screening": "제한상영가",
        "제한상영가": "제한상영가",
    }

    # US 관람등급 → 한국어 매핑 (fallback용)
    cert_us_map: dict[str, str] = {
        "G": "전체 관람가",
        "PG": "전체 관람가",
        "PG-13": "12세 이상 관람가",
        "R": "청소년 관람불가",
        "NC-17": "청소년 관람불가",
        "NR": "",
    }

    def _find_cert(country_code: str, mapping: dict[str, str]) -> str:
        """특정 국가의 관람등급을 찾아 매핑한다."""
        for entry in raw_release_dates:
            if entry.get("iso_3166_1") == country_code:
                releases = entry.get("release_dates", [])
                for release in releases:
                    cert = release.get("certification", "").strip()
                    if cert and cert in mapping:
                        return mapping[cert]
        return ""

    # 1차: KR 관람등급
    result = _find_cert("KR", cert_kr_map)
    if result:
        return result

    # 2차: US 관람등급 (fallback)
    return _find_cert("US", cert_us_map)


def build_embedding_text(doc: MovieDocument) -> str:
    """
    임베딩 모델 입력용 구조화 텍스트를 생성한다.

    Phase A 개선:
    - overview 200자 제한 해제 (임베딩 모델이 긴 텍스트 처리 가능)
    - cast (출연진) 추가: 배우 이름이 검색에 중요
    - certification (관람등급) 추가
    - reviews[0][:300] (첫 번째 리뷰 300자) 추가: 영화 평가 맥락 제공

    Phase ML (다국어 검색 개선):
    - title_en (영문 제목) 추가: 한글 제목과 다를 경우 벡터 공간에 영문 의미도 반영
    - overview_en (영문 줄거리) 추가: 한글 줄거리가 빈약한 경우 영문 줄거리로 보강
      → Upstage Solar 임베딩 모델의 다국어 지원 특성을 활용하여
        영문 쿼리 ↔ 한글+영문 혼합 임베딩 간 벡터 유사도 매칭 개선

    형식: [제목] {title} [영문제목] {title_en} [장르] {genres} [감독] {director}
          [출연] {cast} [키워드] {keywords} [무드] {mood_tags} [관람등급] {certification}
          [줄거리] {overview} [영문줄거리] {overview_en} [리뷰] {reviews[0][:300]}
    """
    parts = [
        f"[제목] {doc.title}",
    ]
    # Phase ML: 영문 제목이 있고 한글 제목과 다를 경우 추가
    # → 벡터 공간에 영문 의미도 반영되어 영문 쿼리 매칭 개선
    if doc.title_en and doc.title_en != doc.title:
        parts.append(f"[영문제목] {doc.title_en}")

    parts.append(f"[장르] {', '.join(doc.genres)}")

    # Phase B: 태그라인 추가 (캐치프레이즈가 있으면 임베딩에 반영)
    if doc.tagline:
        parts.append(f"[태그라인] {doc.tagline}")
    if doc.director:
        parts.append(f"[감독] {doc.director}")
    # Phase ML-2: 감독 원어 이름이 있고 지역화 이름과 다르면 추가
    # → "크리스토퍼 놀란" + "Christopher Nolan" 양방향 임베딩 매칭
    if doc.director_original_name and doc.director_original_name != doc.director:
        parts.append(f"[감독원어] {doc.director_original_name}")
    if doc.cast:
        # Phase ML-2: cast에 이미 한영 이름이 모두 포함됨 (extract_cast_names)
        parts.append(f"[출연] {', '.join(doc.cast)}")
    # Phase B: 확장 크레딧 추가
    if doc.screenwriters:
        parts.append(f"[각본] {', '.join(doc.screenwriters)}")
    if doc.composer:
        parts.append(f"[음악] {doc.composer}")
    if doc.keywords:
        parts.append(f"[키워드] {', '.join(doc.keywords[:10])}")
    if doc.mood_tags:
        parts.append(f"[무드] {', '.join(doc.mood_tags)}")
    if doc.certification:
        parts.append(f"[관람등급] {doc.certification}")
    if doc.overview:
        # Phase A: 200자 제한 해제 → 전체 줄거리 사용
        parts.append(f"[줄거리] {doc.overview}")
    # Phase ML: 영문 줄거리 보강 — 한글 줄거리가 없거나 빈약한(50자 미만) 경우
    # 영문 줄거리를 500자까지 포함하여 임베딩 품질 보완
    if doc.overview_en and (not doc.overview or len(doc.overview) < 50):
        parts.append(f"[영문줄거리] {doc.overview_en[:500]}")
    if doc.reviews:
        # Phase A: 첫 번째 리뷰 300자 추가 (영화 평가 맥락)
        parts.append(f"[리뷰] {doc.reviews[0][:300]}")

    return " ".join(parts)


def get_fallback_mood_tags(genres: list[str]) -> list[str]:
    """
    장르 기반 기본 무드태그를 생성한다 (GPT 실패 시 fallback).

    §11-6-1: 장르 → 무드 기본 매핑 테이블에서 합집합, 중복 제거, 최대 5개.
    """
    mood_set: set[str] = set()
    for genre in genres:
        mood_set.update(GENRE_TO_DEFAULT_MOOD.get(genre, []))

    # 태그 0개인 경우 기본값 (§11-6)
    if not mood_set:
        return ["잔잔"]

    return list(mood_set)[:5]


# ============================================================
# Ollama (qwen2.5:14b) 무드태그 생성
# ============================================================

# §11-10 무드태그 생성 프롬프트
MOOD_TAG_PROMPT = """당신은 영화 분위기 분석 전문가입니다.
다음 영화 정보를 보고, 이 영화의 분위기를 나타내는 무드 태그를 3~5개 생성해주세요.

[영화 정보]
제목: {title}
장르: {genres}
키워드: {keywords}
줄거리: {overview}

[사용 가능한 무드 태그 (이 목록에서만 선택)]
몰입, 감동, 웅장, 긴장감, 힐링, 유쾌, 따뜻, 슬픔, 공포, 잔잔,
스릴, 카타르시스, 청춘, 우정, 가족애, 로맨틱, 미스터리, 반전,
철학적, 사회비판, 모험, 판타지, 레트로, 다크, 유머

JSON 배열로만 응답해주세요. 예: ["몰입", "감동", "웅장"]"""


async def generate_mood_tags(
    title: str,
    genres: list[str],
    keywords: list[str],
    overview: str,
) -> list[str]:
    """
    Ollama (qwen2.5:14b)로 무드태그를 생성한다.

    §11-6 단계 2: 장르+키워드+줄거리 → 3~5개 무드태그.
    §11-6 검증: 화이트리스트 필터링, 실패 시 1회 재시도 후 fallback.

    Ollama는 OpenAI 호환 API를 제공하므로 AsyncOpenAI 클라이언트를 사용한다.
    """
    from openai import AsyncOpenAI

    prompt = MOOD_TAG_PROMPT.format(
        title=title,
        genres=", ".join(genres),
        keywords=", ".join(keywords[:10]),
        overview=overview[:200],
    )

    # Ollama OpenAI 호환 API 사용 (무드태그 생성은 Qwen 모델)
    client = AsyncOpenAI(
        base_url=f"{settings.OLLAMA_BASE_URL}/v1",
        api_key="ollama",
    )

    for attempt in range(2):  # 최대 2회 시도 (1회 재시도)
        try:
            response = await client.chat.completions.create(
                model=settings.MOOD_MODEL,  # qwen2.5:14b
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            content = response.choices[0].message.content or "[]"

            # JSON 배열 부분만 추출 (모델이 부가 텍스트를 출력할 수 있음)
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                content = content[start:end]

            tags = json.loads(content)

            # 화이트리스트 필터링 (§11-6)
            valid_tags = [tag for tag in tags if tag in MOOD_WHITELIST]

            if valid_tags:
                return valid_tags[:5]

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("mood_tag_generation_failed", title=title, attempt=attempt + 1, error=str(e))

    # 모든 시도 실패 시 fallback (§11-6)
    return get_fallback_mood_tags(genres)


# ============================================================
# 데이터 검증 (§11-3)
# ============================================================

CURRENT_YEAR = datetime.now().year


def validate_movie(doc: MovieDocument) -> bool:
    """
    영화 문서의 유효성을 검증한다.

    §11-3 데이터 검증 규칙:
    - 필수 필드: id, title, genres, release_year
    - 평점 범위: 0~10
    - 개봉년도: 1900~현재
    """
    # 필수 필드 존재
    if not doc.id or not doc.title or not doc.genres or not doc.release_year:
        return False

    # 평점 범위
    if doc.rating < 0 or doc.rating > 10:
        return False

    # 개봉년도 범위
    if doc.release_year < 1900 or doc.release_year > CURRENT_YEAR:
        return False

    return True


# ============================================================
# TMDBRawMovie → MovieDocument 변환
# ============================================================


async def process_raw_movie(raw: TMDBRawMovie, generate_mood: bool = True) -> MovieDocument | None:
    """
    TMDBRawMovie를 MovieDocument로 변환한다 (전체 전처리 파이프라인).

    1. 장르 한국어 변환
    2. 감독/배우 추출
    3. 키워드 추출
    4. OTT 정규화
    5. 무드태그 생성 (Ollama qwen2.5:14b)
    6. 임베딩 텍스트 구성
    7. 유효성 검증

    Args:
        raw: TMDB API 원본 데이터
        generate_mood: True이면 Ollama로 무드태그 생성, False이면 fallback 사용

    Returns:
        MovieDocument 또는 None (검증 실패 시)
    """
    # 개봉 연도 추출
    release_year = 0
    if raw.release_date and len(raw.release_date) >= 4:
        try:
            release_year = int(raw.release_date[:4])
        except ValueError:
            pass

    # 장르 변환
    genres = convert_genres(raw.genres)

    # 인물 추출
    director = extract_director(raw.credits)
    cast = extract_cast(raw.credits)

    # Phase ML-2: 감독/배우 한영 이중 이름 추출
    # name(지역화)과 original_name(원어)을 모두 추출하여
    # "크리스토퍼 놀란" / "Christopher Nolan" 양방향 검색 지원
    director_name, director_original_name_extracted = extract_director_names(raw.credits)
    cast_all_names = extract_cast_names(raw.credits)

    # 키워드 추출 (Phase ML-2: 한국어 매핑 포함)
    keywords = extract_keywords(raw.keywords)

    # OTT 정규화
    ott_platforms = normalize_ott_platforms(raw.watch_providers)

    # Phase A: TMDB 보강 데이터 추출
    reviews = extract_reviews(raw.reviews)
    trailer_url = extract_trailer_url(raw.videos)
    behind_the_scenes = extract_behind_the_scenes(raw.videos)
    certification = extract_certification(raw.release_dates)
    similar_movie_ids = [str(mid) for mid in raw.similar_movie_ids]

    # Phase B: 확장 크레딧 추출
    cast_characters = extract_cast_with_characters(raw.credits)
    cinematographer = extract_cinematographer(raw.credits)
    composer = extract_composer(raw.credits)
    screenwriters = extract_screenwriters(raw.credits)
    producers = extract_producers(raw.credits)
    editor_name = extract_editor(raw.credits)

    # Phase C: 감독 상세 정보 + 추가 크루
    director_details = extract_director_details(raw.credits)
    executive_prods = extract_executive_producers(raw.credits)
    prod_designer = extract_production_designer(raw.credits)
    costume_des = extract_costume_designer(raw.credits)
    src_author = extract_source_author(raw.credits)

    # Phase B: 컬렉션/제작사/국가/언어 추출
    collection_id, collection_name = extract_collection(raw.belongs_to_collection)
    production_countries = extract_production_countries(raw.production_countries)
    spoken_languages = extract_spoken_languages(raw.spoken_languages)

    # Phase C: 제작사 확장 (logo_path, origin_country 포함)
    production_companies = extract_production_companies_full(raw.production_companies)

    # Phase C: 국가/언어 전체 이름
    production_country_names = extract_production_country_names(raw.production_countries)
    spoken_language_names = extract_spoken_language_names(raw.spoken_languages)

    # Phase C: 컬렉션 이미지 / 다중 이미지 / KR 개봉일
    collection_poster, collection_backdrop = extract_collection_images(raw.belongs_to_collection)
    images_posters, images_backdrops = extract_images(raw.images)
    kr_release_date = extract_kr_release_date(raw.release_dates)
    # Phase D: recommendations가 list[dict]로 변경 → ID 추출 시 dict에서 "id" 키 사용
    recommendation_ids = [
        str(m.get("id") if isinstance(m, dict) else m)
        for m in raw.recommendations
        if (m.get("id") if isinstance(m, dict) else m)
    ]

    # Phase D: translations에서 다국어 줄거리 추출 + overview 보강
    boosted_overview, overview_en, overview_ja = extract_overview_from_translations(
        raw.translations, raw.overview,
    )

    # Phase D: external_ids에서 소셜 미디어 ID 추출
    ext_ids = extract_external_ids_full(raw.external_ids)

    # Phase D: images.logos 추출
    images_logos = extract_images_logos(raw.images)

    # Phase D: TMDB 사용자 리스트 수
    tmdb_list_count = extract_tmdb_list_count(raw.lists)

    # MovieDocument 생성 (무드태그/임베딩텍스트 제외)
    doc = MovieDocument(
        id=str(raw.id),
        title=raw.title or raw.original_title,
        title_en=raw.original_title,
        overview=boosted_overview,  # Phase D: translations에서 보강된 overview
        release_year=release_year,
        runtime=raw.runtime or 0,
        rating=raw.vote_average,
        vote_count=raw.vote_count,
        popularity_score=raw.popularity,
        poster_path=raw.poster_path or "",
        genres=genres,
        keywords=keywords,
        director=director,
        # Phase ML-2: cast에 한영 이름을 모두 포함하여 양방향 검색 지원
        # 예: ["톰 크루즈", "Tom Cruise", "마일스 텔러", "Miles Teller"]
        # 기존 cast(name만)와 cast_all_names(name+original_name) 중 이중 이름 버전 사용
        cast=cast_all_names if cast_all_names else cast,
        ott_platforms=ott_platforms,
        # Phase A: 보강 필드
        reviews=reviews,
        trailer_url=trailer_url,
        behind_the_scenes=behind_the_scenes,
        certification=certification,
        similar_movie_ids=similar_movie_ids,
        # Phase B: 재무/텍스트 메타데이터
        budget=raw.budget,
        revenue=raw.revenue,
        tagline=raw.tagline,
        homepage=raw.homepage,
        # Phase B: 컬렉션/제작사
        collection_id=collection_id,
        collection_name=collection_name,
        production_companies=production_companies,
        production_countries=production_countries,
        original_language=raw.original_language,
        spoken_languages=spoken_languages,
        imdb_id=raw.imdb_id,
        backdrop_path=raw.backdrop_path or "",
        adult=raw.adult,
        status=raw.status,
        # Phase B: 확장 크레딧
        cast_characters=cast_characters,
        cinematographer=cinematographer,
        composer=composer,
        screenwriters=screenwriters,
        producers=producers,
        editor=editor_name,
        # Phase C: 완전 데이터 추출
        origin_country=raw.origin_country,
        director_id=director_details["id"],
        director_profile_path=director_details["profile_path"],
        director_original_name=director_details["original_name"],
        alternative_titles=raw.alternative_titles,
        recommendation_ids=recommendation_ids,
        images_posters=images_posters,
        images_backdrops=images_backdrops,
        collection_poster_path=collection_poster,
        collection_backdrop_path=collection_backdrop,
        kr_release_date=kr_release_date,
        executive_producers=executive_prods,
        production_designer=prod_designer,
        costume_designer=costume_des,
        source_author=src_author,
        production_country_names=production_country_names,
        spoken_language_names=spoken_language_names,
        # Phase D: 전체 수집 보강 필드
        video_flag=raw.video,
        overview_en=overview_en,
        overview_ja=overview_ja,
        facebook_id=ext_ids["facebook_id"],
        instagram_id=ext_ids["instagram_id"],
        twitter_id=ext_ids["twitter_id"],
        wikidata_id=ext_ids["wikidata_id"],
        tmdb_list_count=tmdb_list_count,
        images_logos=images_logos,
        source="tmdb",
    )

    # 무드태그 생성 (Ollama Qwen 모델 사용)
    if generate_mood and settings.OLLAMA_BASE_URL:
        doc.mood_tags = await generate_mood_tags(doc.title, genres, keywords, doc.overview)
    else:
        doc.mood_tags = get_fallback_mood_tags(genres)

    # 줄거리 빈 문자열 대체 (§11-3)
    if not doc.overview or len(doc.overview) < 10:
        doc.overview = f"{', '.join(genres)} 장르의 영화. 키워드: {', '.join(keywords[:5])}"

    # 임베딩 텍스트 구성
    doc.embedding_text = build_embedding_text(doc)

    # 유효성 검증 (§11-3)
    if not validate_movie(doc):
        logger.warning("movie_validation_failed", id=doc.id, title=doc.title)
        return None

    return doc
