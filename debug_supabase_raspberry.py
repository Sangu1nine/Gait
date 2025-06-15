# 라즈베리파이 Supabase 연결 문제 진단 스크립트
# 작성일: 2024-01-23
# 용도: 라즈베리파이에서 Supabase 연결 문제를 단계별로 진단

import os
import sys
from dotenv import load_dotenv
import subprocess
import socket

def check_environment():
    """환경 확인"""
    print("🔍 시스템 환경 확인")
    print("-" * 50)
    print(f"Python 버전: {sys.version}")
    print(f"작업 디렉토리: {os.getcwd()}")
    print(f"스크립트 경로: {os.path.abspath(__file__)}")
    print()

def check_dotenv_file():
    """환경변수 파일 확인"""
    print("📄 .env 파일 확인")
    print("-" * 50)
    
    env_files = [".env", "../.env", "Gait/.env"]
    
    for env_path in env_files:
        if os.path.exists(env_path):
            print(f"✅ .env 파일 발견: {os.path.abspath(env_path)}")
            
            try:
                with open(env_path, 'r') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    
                print(f"   파일 크기: {len(content)} bytes")
                print(f"   줄 수: {len(lines)}")
                
                # 키 존재 여부 확인 (값은 보안상 표시하지 않음)
                has_url = any('SUPABASE_URL' in line for line in lines)
                has_key = any('SUPABASE_KEY' in line for line in lines)
                
                print(f"   SUPABASE_URL 포함: {'✅' if has_url else '❌'}")
                print(f"   SUPABASE_KEY 포함: {'✅' if has_key else '❌'}")
                
                if has_url and has_key:
                    print("✅ 필요한 환경변수가 .env 파일에 있습니다")
                    return env_path
                    
            except Exception as e:
                print(f"❌ .env 파일 읽기 실패: {e}")
        else:
            print(f"❌ .env 파일 없음: {os.path.abspath(env_path)}")
    
    print("❌ 유효한 .env 파일을 찾을 수 없습니다")
    return None

def test_dotenv_loading(env_path):
    """환경변수 로딩 테스트"""
    print("\n🔑 환경변수 로딩 테스트")
    print("-" * 50)
    
    try:
        # 특정 경로에서 로드
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        print(f"SUPABASE_URL: {'✅ 로드됨' if url else '❌ 없음'}")
        if url:
            print(f"   URL: {url}")
            
        print(f"SUPABASE_KEY: {'✅ 로드됨' if key else '❌ 없음'}")
        if key:
            print(f"   Key prefix: {key[:20]}...")
            print(f"   Key length: {len(key)} characters")
        
        return url, key
        
    except Exception as e:
        print(f"❌ 환경변수 로딩 실패: {e}")
        return None, None

def check_network_connectivity():
    """네트워크 연결성 확인"""
    print("\n🌐 네트워크 연결성 확인")
    print("-" * 50)
    
    # 인터넷 연결 확인
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("✅ 인터넷 연결 정상")
    except OSError:
        print("❌ 인터넷 연결 실패")
        return False
    
    # DNS 해상도 확인
    try:
        socket.gethostbyname("supabase.co")
        print("✅ DNS 해상도 정상")
    except socket.gaierror:
        print("❌ DNS 해상도 실패")
        return False
    
    return True

def check_dependencies():
    """필요한 라이브러리 확인"""
    print("\n📦 의존성 라이브러리 확인")
    print("-" * 50)
    
    required_packages = [
        ('python-dotenv', 'dotenv'),
        ('supabase', 'supabase'),
        ('requests', 'requests')
    ]
    
    all_installed = True
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}: 설치됨")
        except ImportError:
            print(f"❌ {package_name}: 설치되지 않음")
            all_installed = False
    
    if not all_installed:
        print("\n설치 명령어:")
        print("pip install python-dotenv supabase requests")
    
    return all_installed

def test_supabase_connection(url, key):
    """Supabase 연결 테스트"""
    print("\n🔗 Supabase 연결 테스트")
    print("-" * 50)
    
    if not url or not key:
        print("❌ URL 또는 키가 없어서 연결 테스트를 할 수 없습니다")
        return False
    
    try:
        # requests로 기본 연결 테스트
        import requests
        
        print("1️⃣ REST API 기본 연결 테스트...")
        response = requests.get(
            f"{url}/rest/v1/",
            headers={"apikey": key},
            timeout=10
        )
        print(f"   HTTP 상태: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ REST API 연결 성공")
        elif response.status_code == 401:
            print("❌ 인증 실패 - API 키를 확인하세요")
            return False
        elif response.status_code == 404:
            print("❌ 프로젝트 URL이 올바르지 않습니다")
            return False
        
    except requests.exceptions.ConnectionError:
        print("❌ 연결 실패 - 네트워크를 확인하세요")
        return False
    except requests.exceptions.Timeout:
        print("❌ 연결 시간 초과")
        return False
    except Exception as e:
        print(f"❌ 연결 테스트 실패: {e}")
        return False
    
    try:
        # supabase-py 라이브러리 테스트
        print("2️⃣ supabase-py 라이브러리 테스트...")
        from supabase import create_client
        
        client = create_client(url, key)
        print("✅ Supabase 클라이언트 생성 성공")
        
        # 간단한 쿼리 테스트
        print("3️⃣ 데이터베이스 접근 테스트...")
        try:
            response = client.table('fall_history').select('count').limit(1).execute()
            print("✅ 데이터베이스 접근 성공")
        except Exception as db_error:
            print(f"⚠️ 데이터베이스 접근 테스트: {db_error}")
            print("   (테이블이 없을 수 있지만 연결은 정상일 수 있습니다)")
        
        return True
        
    except Exception as e:
        print(f"❌ supabase-py 테스트 실패: {e}")
        return False

def main():
    """메인 진단 함수"""
    print("🔧 라즈베리파이 Supabase 연결 진단 도구")
    print("=" * 60)
    
    # 1. 환경 확인
    check_environment()
    
    # 2. .env 파일 확인
    env_path = check_dotenv_file()
    
    # 3. 환경변수 로딩 테스트
    url, key = test_dotenv_loading(env_path)
    
    # 4. 네트워크 연결성 확인
    if not check_network_connectivity():
        print("\n❌ 네트워크 문제로 인해 더 이상 진행할 수 없습니다")
        return
    
    # 5. 의존성 확인
    if not check_dependencies():
        print("\n❌ 필요한 라이브러리가 설치되지 않았습니다")
        return
    
    # 6. Supabase 연결 테스트
    if test_supabase_connection(url, key):
        print("\n✅ 모든 테스트 통과! Supabase 연결이 정상입니다")
    else:
        print("\n❌ Supabase 연결에 문제가 있습니다")
    
    print("\n" + "=" * 60)
    print("진단 완료")

if __name__ == "__main__":
    main() 