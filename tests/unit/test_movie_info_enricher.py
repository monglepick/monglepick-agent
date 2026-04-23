"""
movie_info_enricher 단위 테스트 (2026-04-23 추가).

대상 모듈: monglepick.utils.movie_info_enricher
검증 범위:
 1) _needs_enrichment / _extract_useful_text / _build_search_query 순수 함수
 2) _merge_search_results 의 도메인 우선순위 + 중복 제거
 3) enrich_movie_overview 의 캐시 동작 (DDGS 는 monkeypatch)
 4) enrich_movies_batch 의 선택적 보강 (이미 긴 overview 는 스킵)
 5) 외부 검색 신규 유틸: _build_external_query / _extract_movie_candidates /
    search_external_movies (DDGS 는 monkeypatch)

DuckDuckGo 네트워크 호출은 전부 monkeypatch 로 차단한다.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from monglepick.utils import movie_info_enricher as enricher


# ============================================================
# 1. 순수 함수들
# ============================================================


class TestNeedsEnrichment:
    """_needs_enrichment: overview 길이 판정."""

    def test_none_returns_true(self):
        assert enricher._needs_enrichment(None) is True

    def test_empty_returns_true(self):
        assert enricher._needs_enrichment("") is True

    def test_short_returns_true(self):
        # 250자 미만
        assert enricher._needs_enrichment("짧은 줄거리") is True

    def test_long_returns_false(self):
        # 250자 이상이면 보강 불필요
        long_text = "가" * 260
        assert enricher._needs_enrichment(long_text) is False


class TestExtractUsefulText:
    """_extract_useful_text: HTML 정제 + 최소 길이 필터."""

    def test_strips_html_tags(self):
        # NOTE: _extract_useful_text 는 20자 미만이면 빈 문자열을 반환하므로
        # 태그 제거 후에도 20자 이상이 되도록 충분한 길이의 본문으로 검증.
        raw = "<p>봉준호 <b>감독의</b> 스릴러 영화로 2006년에 개봉한 한국 영화입니다</p>"
        out = enricher._extract_useful_text(raw)
        assert "<" not in out
        assert ">" not in out
        assert "봉준호" in out

    def test_collapses_whitespace(self):
        raw = "줄거리   \n\n  여러 줄   공백"
        out = enricher._extract_useful_text(raw)
        assert "  " not in out
        assert "\n" not in out

    def test_too_short_returns_empty(self):
        assert enricher._extract_useful_text("<p>짧</p>") == ""

    def test_empty_input(self):
        assert enricher._extract_useful_text("") == ""


class TestBuildSearchQuery:
    """_build_search_query: 영화명/영문명/연도/키워드 조립."""

    def test_minimum_query_has_title_and_keyword(self):
        q = enricher._build_search_query("인터스텔라")
        assert "인터스텔라" in q
        assert "영화 줄거리 시놉시스" in q

    def test_includes_english_title_when_different(self):
        q = enricher._build_search_query("인터스텔라", title_en="Interstellar")
        assert "Interstellar" in q

    def test_skips_english_title_when_same(self):
        q = enricher._build_search_query("Interstellar", title_en="Interstellar")
        # 중복 방지: 한국어 제목과 영문 제목이 같으면 한 번만 들어가야 한다
        assert q.count("Interstellar") == 1

    def test_includes_release_year(self):
        q = enricher._build_search_query("괴물", release_year=2006)
        assert "2006" in q


class TestBuildExternalQuery:
    """_build_external_query: external_search_node 전용 쿼리."""

    def test_uses_user_intent_when_available(self):
        q = enricher._build_external_query(
            user_intent="따뜻한 가족 영화",
            current_input="뭐 볼까",
        )
        assert "따뜻한 가족 영화" in q
        # current_input 은 fallback 이므로 user_intent 가 있으면 쓰지 않는다
        assert "뭐 볼까" not in q

    def test_falls_back_to_current_input(self):
        q = enricher._build_external_query(
            user_intent="",
            current_input="2026년 영화",
        )
        assert "2026년 영화" in q

    def test_adds_year_when_recency_signal_present(self):
        q = enricher._build_external_query(
            user_intent="최신 영화",
            current_input="",
            release_year_gte=2026,
        )
        assert "2026년" in q

    def test_default_when_all_empty(self):
        q = enricher._build_external_query("", "")
        # 완전히 빈 입력이어도 최소한의 검색어는 나와야 한다
        assert len(q) > 0


# ============================================================
# 2. 결과 병합 (도메인 우선순위)
# ============================================================


class TestMergeSearchResults:
    """_merge_search_results: Wikipedia/나무위키 우선 + 중복 제거."""

    def test_empty_returns_empty(self):
        assert enricher._merge_search_results([]) == ""

    def test_wikipedia_takes_priority_over_random_blog(self):
        # 같은 내용이 두 소스에 있고 앞 30자가 동일 → 우선순위 높은 소스만 남는다
        results = [
            {
                "title": "개인 블로그",
                "body": "봉준호 감독의 괴물 은 2006년 개봉한 한국 영화입니다 긴 본문 본문 본문 본문.",
                "href": "https://some-blog.com/post/1",
            },
            {
                "title": "괴물 - 위키백과",
                "body": "봉준호 감독의 괴물 은 2006년 개봉한 한국 영화로 한강에서 벌어지는 이야기입니다 본문 본문.",
                "href": "https://ko.wikipedia.org/wiki/괴물",
            },
        ]
        merged = enricher._merge_search_results(results)
        # 앞 30자가 동일하므로 중복 필터로 블로그가 드롭되고 위키만 남는다
        # 위키 본문의 "한강" 키워드가 들어있어야 한다
        assert "한강" in merged

    def test_truncates_to_500_chars(self):
        long_body = "줄거리 " * 300  # 1500자 이상
        results = [{"title": "T", "body": long_body, "href": "https://ko.wikipedia.org/x"}]
        merged = enricher._merge_search_results(results)
        assert len(merged) <= 500


# ============================================================
# 3. enrich_movie_overview (캐시 + 보강 스킵)
# ============================================================


@pytest.fixture(autouse=True)
def _clear_cache():
    """매 테스트마다 인메모리 캐시 초기화 — 테스트 간 간섭 방지."""
    enricher.clear_enrichment_cache()
    yield
    enricher.clear_enrichment_cache()


class TestEnrichMovieOverview:
    """enrich_movie_overview: 캐시 + 보강 스킵 로직."""

    @pytest.mark.asyncio
    async def test_skips_if_overview_already_long(self):
        """overview 가 이미 250자 이상이면 DDGS 호출이 일어나선 안 된다."""
        long_overview = "가" * 300
        movie = {"title": "테스트", "overview": long_overview}

        with patch.object(enricher, "_search_duckduckgo") as mock_ddg:
            out = await enricher.enrich_movie_overview(movie)

        mock_ddg.assert_not_called()
        assert out is movie  # 원본 그대로 반환

    @pytest.mark.asyncio
    async def test_skips_if_no_title(self):
        """title 이 비어있으면 외부 검색을 시도하지 않는다."""
        movie = {"title": "", "overview": ""}

        with patch.object(enricher, "_search_duckduckgo") as mock_ddg:
            out = await enricher.enrich_movie_overview(movie)

        mock_ddg.assert_not_called()
        assert out is movie

    @pytest.mark.asyncio
    async def test_uses_cache_on_second_call(self):
        """동일 title_year 키로 재호출 시 DDGS 가 다시 호출되지 않는다."""
        movie = {"title": "괴물", "overview": "", "release_year": 2006}
        ddg_results = [{
            "title": "괴물 - 위키",
            "body": "2006년 한국 영화 괴물 입니다 " * 10,  # 충분히 긴 텍스트
            "href": "https://ko.wikipedia.org/wiki/괴물",
        }]

        with patch.object(enricher, "_search_duckduckgo", return_value=ddg_results) as mock_ddg:
            out1 = await enricher.enrich_movie_overview(movie)
            out2 = await enricher.enrich_movie_overview(movie)

        assert mock_ddg.call_count == 1  # 한 번만 실제 검색
        assert out1.get("_enriched") is True
        assert out2.get("_enriched") is True

    @pytest.mark.asyncio
    async def test_graceful_on_empty_results(self):
        """DDGS 가 빈 리스트를 반환해도 원본을 유지하고 에러를 내지 않는다."""
        movie = {"title": "없는영화", "overview": ""}

        with patch.object(enricher, "_search_duckduckgo", return_value=[]):
            out = await enricher.enrich_movie_overview(movie)

        # 보강 실패 → 원본 반환, _enriched 플래그 없음
        assert out is movie or out.get("_enriched") is not True


# ============================================================
# 4. enrich_movies_batch (선택적 보강)
# ============================================================


class TestEnrichMoviesBatch:
    """enrich_movies_batch: 길이 기준 선택적 보강 + 순서 유지."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        out = await enricher.enrich_movies_batch([])
        assert out == []

    @pytest.mark.asyncio
    async def test_skips_movies_with_sufficient_overview(self):
        """모든 영화가 이미 충분한 overview 를 가지면 DDGS 호출 없이 통과."""
        movies = [
            {"title": "A", "overview": "가" * 300},
            {"title": "B", "overview": "나" * 300},
        ]

        with patch.object(enricher, "_search_duckduckgo") as mock_ddg:
            out = await enricher.enrich_movies_batch(movies)

        mock_ddg.assert_not_called()
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """입력 순서가 그대로 출력 순서로 유지되어야 한다."""
        movies = [{"title": f"M{i}", "overview": "가" * 300, "id": str(i)} for i in range(5)]
        out = await enricher.enrich_movies_batch(movies)
        assert [m["id"] for m in out] == ["0", "1", "2", "3", "4"]


# ============================================================
# 5. 외부 검색 신규 유틸 (_extract_movie_candidates, search_external_movies)
# ============================================================


class TestExtractMovieCandidates:
    """_extract_movie_candidates: DDGS 결과 → 영화 스텁 변환."""

    def test_empty_returns_empty(self):
        assert enricher._extract_movie_candidates([]) == []

    def test_extracts_quoted_title_with_year(self):
        """「제목」(YYYY) 패턴에서 제목과 연도를 추출한다."""
        results = [{
            "title": "2026년 영화 - 나무위키",
            "body": "「괴물 리턴즈」(2026) 봉준호 감독의 신작 영화입니다.",
            "href": "https://namu.wiki/w/2026",
        }]
        extracted = enricher._extract_movie_candidates(results)
        assert len(extracted) == 1
        assert extracted[0]["title"] == "괴물 리턴즈"
        assert extracted[0]["release_year"] == 2026
        assert extracted[0]["_external"] is True
        assert extracted[0]["id"].startswith("external_")

    def test_deduplicates_by_title(self):
        """동일 제목(대소문자/공백 무시) 결과는 한 번만 추출."""
        results = [
            {"title": "A", "body": "「인터스텔라」(2014) 놀란 감독.", "href": "https://ko.wikipedia.org/x"},
            {"title": "B", "body": "「인터스텔라」(2014) 다른 소스.", "href": "https://namu.wiki/y"},
        ]
        extracted = enricher._extract_movie_candidates(results)
        # 중복 제거 후 한 개만 남는다
        assert len(extracted) == 1

    def test_respects_max_movies_limit(self):
        """max_movies 를 초과해서 추출하지 않는다."""
        results = [
            {"title": f"T{i}", "body": f"「영화{i}」(2026) 설명", "href": f"https://x.com/{i}"}
            for i in range(10)
        ]
        extracted = enricher._extract_movie_candidates(results, max_movies=3)
        assert len(extracted) == 3

    def test_prioritizes_trusted_domains(self):
        """Wikipedia/나무위키 결과가 먼저 배치되어 먼저 추출된다."""
        results = [
            {"title": "블로그", "body": "「영화B」(2026) 블로그", "href": "https://blog.com/1"},
            {"title": "위키", "body": "「영화A」(2026) 위키", "href": "https://ko.wikipedia.org/A"},
        ]
        extracted = enricher._extract_movie_candidates(results, max_movies=2)
        # 위키 (_PRIORITY_DOMAINS 내) 가 블로그 (priority len) 보다 먼저
        assert extracted[0]["title"] == "영화A"

    def test_kmdb_outranks_wikipedia(self):
        """2026-04-23 후속: KMDB 가 Wikipedia 보다 우선순위가 높다 (한국 영화 정확도)."""
        results = [
            {
                "title": "영화 정보 - 위키피디아",
                "body": "「괴물」(2006) 한국 영화 위키피디아 설명",
                "href": "https://ko.wikipedia.org/wiki/괴물",
            },
            {
                "title": "영화 정보 - KMDB",
                "body": "「마더」(2009) 봉준호 감독 한국영상자료원 공식",
                "href": "https://www.kmdb.or.kr/movies/111",
            },
        ]
        extracted = enricher._extract_movie_candidates(results, max_movies=2)
        # KMDB (priority 0) 가 Wikipedia (priority 2) 보다 먼저
        assert extracted[0]["title"] == "마더"
        assert extracted[1]["title"] == "괴물"

    def test_priority_domains_is_module_level_constant(self):
        """2026-04-23 후속: _PRIORITY_DOMAINS 가 모듈 상수로 존재(단일 진실 원본)."""
        assert hasattr(enricher, "_PRIORITY_DOMAINS")
        # KMDB/KOBIS 가 가장 앞쪽
        assert enricher._PRIORITY_DOMAINS[0] == "kmdb.or.kr"
        assert enricher._PRIORITY_DOMAINS[1] == "kobis.or.kr"


# ============================================================
# 6. 매칭 검증 (2026-04-23 후속 P1 — 오보강 방지)
# ============================================================


class TestIsValidMovieTitle:
    """_is_valid_movie_title: 제네릭/무효 제목 드롭."""

    def test_rejects_empty_title(self):
        assert enricher._is_valid_movie_title("") is False
        assert enricher._is_valid_movie_title(None) is False  # type: ignore[arg-type]

    def test_rejects_too_short_title(self):
        assert enricher._is_valid_movie_title("가") is False
        assert enricher._is_valid_movie_title("a") is False

    @pytest.mark.parametrize("generic", ["영화", "최신 영화", "추천", "개봉 영화", "OTT"])
    def test_rejects_generic_blacklist(self, generic):
        """블랙리스트 단어는 공백/대소문자 무시하고 드롭."""
        assert enricher._is_valid_movie_title(generic) is False

    def test_rejects_number_only(self):
        """숫자나 특수문자만 있는 제목은 영화가 아님."""
        assert enricher._is_valid_movie_title("2026") is False
        assert enricher._is_valid_movie_title("!!!") is False
        assert enricher._is_valid_movie_title("12-34") is False

    def test_accepts_valid_korean_title(self):
        assert enricher._is_valid_movie_title("괴물") is True
        assert enricher._is_valid_movie_title("기생충") is True

    def test_accepts_valid_english_title(self):
        assert enricher._is_valid_movie_title("Interstellar") is True

    def test_accepts_mixed_title(self):
        assert enricher._is_valid_movie_title("스파이더맨: 홈커밍") is True


class TestIsYearCompatible:
    """_is_year_compatible: 요청 연도와 호환성 판정."""

    def test_none_gte_always_compatible(self):
        """release_year_gte 가 없으면 모든 연도 허용."""
        assert enricher._is_year_compatible(2000, None) is True
        assert enricher._is_year_compatible(0, None) is True

    def test_zero_extracted_year_allowed(self):
        """연도 미상(0) 은 제목 기반 판단을 위해 통과."""
        assert enricher._is_year_compatible(0, 2026) is True

    def test_within_tolerance_allowed(self):
        """tolerance(3년) 이내는 통과."""
        assert enricher._is_year_compatible(2023, 2026) is True  # -3 경계
        assert enricher._is_year_compatible(2024, 2026) is True

    def test_outside_tolerance_rejected(self):
        """tolerance(3년) 초과는 드롭."""
        assert enricher._is_year_compatible(2020, 2026) is False
        assert enricher._is_year_compatible(2010, 2026) is False

    def test_future_year_allowed(self):
        """하한만 검사 — 요청 연도보다 미래는 항상 통과."""
        assert enricher._is_year_compatible(2030, 2026) is True


class TestExtractMovieCandidatesValidation:
    """_extract_movie_candidates: 매칭 검증 통합."""

    def test_drops_generic_titles(self):
        """「영화」(2026) 같은 제네릭 제목은 드롭."""
        results = [
            {"title": "검색결과", "body": "「영화」(2026) 일반 명사", "href": "https://x.com/1"},
            {"title": "검색결과", "body": "「괴물 리턴즈」(2026) 봉준호 신작", "href": "https://x.com/2"},
        ]
        extracted = enricher._extract_movie_candidates(results, max_movies=5)
        # "영화" 는 블랙리스트에 걸려 드롭, "괴물 리턴즈" 만 남는다
        assert len(extracted) == 1
        assert extracted[0]["title"] == "괴물 리턴즈"

    def test_drops_year_mismatched_movies(self):
        """release_year_gte=2026 인데 추출 연도가 2010 이면 드롭."""
        results = [
            {"title": "옛날영화", "body": "「기생충」(2019) 봉준호", "href": "https://x.com/1"},
            {"title": "신작", "body": "「파묘2」(2026) 장재현", "href": "https://x.com/2"},
        ]
        extracted = enricher._extract_movie_candidates(
            results, max_movies=5, release_year_gte=2026,
        )
        # 2019 는 2026-3=2023 보다 아래 → 드롭
        assert len(extracted) == 1
        assert extracted[0]["title"] == "파묘2"

    def test_allows_year_missing_when_gte_set(self):
        """연도 미상(0)이어도 제목이 유효하면 통과."""
        # 연도 없이 따옴표로만 제목 추출됨
        results = [
            {"title": "추천", "body": "「괴물 리턴즈」 봉준호 감독", "href": "https://x.com/1"},
        ]
        extracted = enricher._extract_movie_candidates(
            results, max_movies=5, release_year_gte=2026,
        )
        assert len(extracted) == 1
        assert extracted[0]["release_year"] == 0

    def test_validation_pipeline_end_to_end(self):
        """실전 시나리오: 제네릭 + 연도 미스매치 + 유효 영화 혼재."""
        results = [
            {"title": "리스트", "body": "「최신 영화」(2026) 제네릭", "href": "https://x.com/1"},
            {"title": "옛날", "body": "「괴물」(2006) 옛날 봉준호", "href": "https://x.com/2"},
            {"title": "신작", "body": "「괴물 리턴즈」(2026) 신작", "href": "https://kmdb.or.kr/3"},
        ]
        extracted = enricher._extract_movie_candidates(
            results, max_movies=5, release_year_gte=2026,
        )
        titles = [m["title"] for m in extracted]
        assert "괴물 리턴즈" in titles
        assert "최신 영화" not in titles  # 블랙리스트
        assert "괴물" not in titles  # 연도 미스매치


class TestSearchExternalMovies:
    """search_external_movies: DDGS 호출 + 스텁 변환 end-to-end."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_ddg_failure(self):
        """DDGS 가 빈 리스트 반환 시 조용히 빈 리스트로 graceful."""
        with patch.object(enricher, "_search_duckduckgo", return_value=[]):
            out = await enricher.search_external_movies(
                user_intent="최신 영화",
                current_input="",
                release_year_gte=2026,
            )
        assert out == []

    @pytest.mark.asyncio
    async def test_returns_extracted_stubs(self):
        """DDGS 결과가 있으면 스텁 dict 리스트를 반환한다."""
        ddg_results = [{
            "title": "2026년 영화 - 나무위키",
            "body": "「파묘2」(2026) 장재현 감독의 후속편입니다.",
            "href": "https://namu.wiki/w/2026",
        }]
        with patch.object(enricher, "_search_duckduckgo", return_value=ddg_results):
            out = await enricher.search_external_movies(
                user_intent="2026년 최신 영화",
                current_input="",
                release_year_gte=2026,
                max_movies=5,
            )
        assert len(out) == 1
        assert out[0]["title"] == "파묘2"
        assert out[0]["release_year"] == 2026
        assert out[0]["_external"] is True
