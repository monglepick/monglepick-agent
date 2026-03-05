"""
이미지 업로드 보안 강화 테스트.

보안 헬퍼 함수 4개를 단위 테스트한다:
- _strip_base64_prefix: Data URL 접두사 제거 + 패딩 보정
- _validate_image_bytes: JPEG/PNG 매직바이트 검증
- _check_upload_rate_limit: IP당 분당 업로드 횟수 제한
- DecompressionBomb 방어: IMAGE_MAX_PIXELS 설정 확인

테스트 클래스:
- TestStripBase64Prefix (5개)
- TestValidateImageBytes (5개)
- TestCheckUploadRateLimit (4개)
- TestDecompressionBombDefense (1개)
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from monglepick.api.chat import (
    _check_upload_rate_limit,
    _strip_base64_prefix,
    _upload_timestamps,
    _validate_image_bytes,
)


# ============================================================
# Data URL 접두사 제거 + 패딩 보정 테스트
# ============================================================


class TestStripBase64Prefix:
    """_strip_base64_prefix 헬퍼 테스트."""

    def test_jpeg_data_url_prefix_removed(self):
        """JPEG Data URL 접두사가 제거된다."""
        data = "data:image/jpeg;base64,/9j/4AAQ"
        result = _strip_base64_prefix(data)
        assert result == "/9j/4AAQ"

    def test_png_data_url_prefix_removed(self):
        """PNG Data URL 접두사가 제거된다."""
        data = "data:image/png;base64,iVBORw0K"
        result = _strip_base64_prefix(data)
        assert result == "iVBORw0K"

    def test_padding_correction(self):
        """base64 패딩이 4의 배수가 되도록 보정된다."""
        # 길이 5 → 나머지 1 → "===" 추가하여 길이 8로 보정
        data = "AAAAA"
        result = _strip_base64_prefix(data)
        assert len(result) % 4 == 0
        assert result.endswith("===")

    def test_already_valid_base64_passthrough(self):
        """유효한 base64 문자열은 변경 없이 통과한다."""
        data = "/9j/4AAQ"  # 길이 8, 4의 배수
        result = _strip_base64_prefix(data)
        assert result == "/9j/4AAQ"

    def test_empty_string(self):
        """빈 문자열은 그대로 반환된다."""
        result = _strip_base64_prefix("")
        assert result == ""


# ============================================================
# 매직바이트 검증 테스트
# ============================================================


class TestValidateImageBytes:
    """_validate_image_bytes 헬퍼 테스트."""

    def test_jpeg_magic_bytes_pass(self):
        """JPEG 매직바이트(FF D8 FF)가 통과한다."""
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        # 예외 없이 통과해야 함
        _validate_image_bytes(jpeg_bytes)

    def test_png_magic_bytes_pass(self):
        """PNG 매직바이트(89 PNG)가 통과한다."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        _validate_image_bytes(png_bytes)

    def test_gif_magic_bytes_rejected(self):
        """GIF 매직바이트(GIF89a)가 거부된다 (415)."""
        gif_bytes = b"GIF89a" + b"\x00" * 100
        with pytest.raises(ValueError, match="415"):
            _validate_image_bytes(gif_bytes)

    def test_executable_rejected(self):
        """실행 파일 매직바이트(MZ)가 거부된다 (415)."""
        exe_bytes = b"MZ" + b"\x00" * 100
        with pytest.raises(ValueError, match="415"):
            _validate_image_bytes(exe_bytes)

    def test_empty_bytes_rejected(self):
        """빈 바이트가 거부된다 (400)."""
        with pytest.raises(ValueError, match="400"):
            _validate_image_bytes(b"")

    def test_svg_text_rejected(self):
        """SVG 텍스트 데이터가 거부된다 (415)."""
        svg_bytes = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
        with pytest.raises(ValueError, match="415"):
            _validate_image_bytes(svg_bytes)


# ============================================================
# IP당 업로드 Rate Limiting 테스트
# ============================================================


class TestCheckUploadRateLimit:
    """_check_upload_rate_limit 헬퍼 테스트."""

    def setup_method(self):
        """각 테스트 전에 업로드 타임스탬프를 초기화한다."""
        _upload_timestamps.clear()

    def test_under_limit_passes(self):
        """제한 이하의 요청은 통과한다."""
        # 기본 제한: 분당 10회
        for _ in range(5):
            _check_upload_rate_limit("192.168.1.1")
        # 5회 → 통과 (예외 없음)

    def test_over_limit_raises_429(self):
        """제한 초과 시 429 에러가 발생한다."""
        # settings.IMAGE_UPLOAD_RATE_LIMIT 기본값 10 사용
        with patch("monglepick.api.chat.settings") as mock_settings:
            mock_settings.IMAGE_UPLOAD_RATE_LIMIT = 3
            for _ in range(3):
                _check_upload_rate_limit("10.0.0.1")
            # 4번째 요청 → 한도 초과
            with pytest.raises(ValueError, match="429"):
                _check_upload_rate_limit("10.0.0.1")

    def test_expired_timestamps_cleaned(self):
        """60초 이전의 타임스탬프가 자동 만료된다."""
        ip = "10.0.0.2"
        # 61초 전 타임스탬프 삽입 (이미 만료)
        _upload_timestamps[ip] = [time.time() - 61.0] * 5
        # 만료 후 → 제한 이하 → 통과
        _check_upload_rate_limit(ip)
        # 만료된 5개가 제거되고 현재 1개만 남음
        assert len(_upload_timestamps[ip]) == 1

    def test_different_ips_independent(self):
        """서로 다른 IP는 독립적으로 카운트된다."""
        with patch("monglepick.api.chat.settings") as mock_settings:
            mock_settings.IMAGE_UPLOAD_RATE_LIMIT = 2
            # IP-A: 2회 사용 → 한도 도달
            _check_upload_rate_limit("ip-a")
            _check_upload_rate_limit("ip-a")
            # IP-A: 3번째 → 429
            with pytest.raises(ValueError, match="429"):
                _check_upload_rate_limit("ip-a")
            # IP-B: 첫 번째 → 통과 (IP-A와 독립)
            _check_upload_rate_limit("ip-b")


# ============================================================
# DecompressionBomb 방어 테스트
# ============================================================


class TestDecompressionBombDefense:
    """Pillow DecompressionBomb 방어 설정 테스트."""

    def test_max_image_pixels_set_in_resize(self):
        """_resize_image_bytes가 Image.MAX_IMAGE_PIXELS를 설정한다."""
        from PIL import Image as PILImage

        from monglepick.api.chat import _resize_image_bytes
        from monglepick.config import settings

        # 유효한 1x1 JPEG 이미지 생성
        import io
        img = PILImage.new("RGB", (1, 1), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        tiny_jpeg = buf.getvalue()

        # _resize_image_bytes 호출 → MAX_IMAGE_PIXELS가 설정되어야 함
        _resize_image_bytes(tiny_jpeg)
        assert PILImage.MAX_IMAGE_PIXELS == settings.IMAGE_MAX_PIXELS
