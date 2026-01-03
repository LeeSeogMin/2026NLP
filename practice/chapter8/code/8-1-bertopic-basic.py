"""
8-1-bertopic-basic.py
BERTopic 기본 토픽 모델링

이 스크립트는 BERTopic을 사용하여 문서 집합에서 토픽을 자동으로 추출하는
기본적인 워크플로우를 보여준다.

실행 방법:
    python 8-1-bertopic-basic.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

# BERTopic 파이프라인 구성 요소
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# 샘플 문서 데이터 (한국어 뉴스 제목 예시)
documents = [
    # 경제 관련
    "삼성전자 반도체 수출 사상 최대 실적 달성",
    "한국은행 기준금리 동결 결정 배경 분석",
    "코스피 지수 3000선 돌파 투자자 관심 급증",
    "현대차 전기차 생산량 전년 대비 30% 증가",
    "SK하이닉스 HBM 반도체 글로벌 시장 점유율 1위",
    "원달러 환율 1300원대 진입 수출기업 영향",
    "카카오뱅크 대출 금리 인하 소비자 반응",
    "LG에너지솔루션 배터리 수주 역대 최고치",

    # 스포츠 관련
    "손흥민 시즌 15호골 기록 팀 승리 이끌어",
    "프로야구 개막전 관중 10만명 돌파 기대",
    "김연경 은퇴 후 배구 해설위원 활동 시작",
    "이강인 라리가 데뷔골 현지 언론 극찬",
    "한국 축구 대표팀 월드컵 예선 3연승 달성",
    "KBO리그 신인왕 경쟁 치열 주목",
    "LPGA 한국 선수 우승 국내 골프 열기",

    # 기술 관련
    "챗GPT 활용한 업무 자동화 기업들 도입 확대",
    "메타버스 플랫폼 MAU 1억명 돌파 전망",
    "국내 AI 스타트업 투자 유치 1조원 시대",
    "자율주행차 레벨4 상용화 임박 규제 논의",
    "클라우드 컴퓨팅 시장 성장 디지털 전환 가속",
    "애플 비전프로 국내 출시 XR 시장 확대",
    "오픈AI GPT-5 개발 속도 경쟁 심화",

    # 문화 관련
    "BTS 신곡 빌보드 1위 등극 K-POP 위상",
    "넷플릭스 한국 드라마 글로벌 인기 지속",
    "한국 영화 아카데미 후보 2년 연속 진출",
    "뮤지컬 시장 회복세 공연 예매율 상승",
    "K-POP 걸그룹 일본 오리콘 1위 석권",
    "한국 게임 글로벌 매출 순위 상위권 진입",
]


def main():
    """메인 함수: BERTopic 기본 워크플로우 실행"""

    print("=" * 60)
    print("BERTopic 기본 토픽 모델링")
    print("=" * 60)
    print(f"\n총 문서 수: {len(documents)}")

    # Step 1: 임베딩 모델 설정
    # 다국어 지원 모델 사용 (한국어 포함)
    print("\n[1단계] 임베딩 모델 로딩...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("   - 모델: paraphrase-multilingual-MiniLM-L12-v2")
    print("   - 임베딩 차원: 384")

    # Step 2: UMAP 차원 축소 설정
    print("\n[2단계] UMAP 차원 축소 설정...")
    umap_model = UMAP(
        n_neighbors=5,        # 작은 데이터셋에 맞게 조정
        n_components=5,       # 클러스터링용 차원
        min_dist=0.0,         # 밀집된 클러스터 유도
        metric='cosine',      # 코사인 유사도
        random_state=42
    )
    print("   - n_neighbors: 5")
    print("   - n_components: 5")

    # Step 3: HDBSCAN 클러스터링 설정
    print("\n[3단계] HDBSCAN 클러스터링 설정...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,   # 최소 클러스터 크기
        min_samples=2,        # 밀도 추정 샘플
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    print("   - min_cluster_size: 3")
    print("   - min_samples: 2")

    # Step 4: CountVectorizer 설정 (c-TF-IDF용)
    print("\n[4단계] CountVectorizer 설정...")
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),   # 1-gram, 2-gram 포함
        min_df=1,             # 최소 문서 빈도
        max_df=0.95           # 최대 문서 빈도
    )
    print("   - ngram_range: (1, 2)")

    # Step 5: BERTopic 모델 생성 및 학습
    print("\n[5단계] BERTopic 모델 학습...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=5,        # 토픽당 상위 5개 단어
        verbose=False
    )

    # 문서에 대한 토픽 추출
    topics, probs = topic_model.fit_transform(documents)

    print("\n" + "=" * 60)
    print("토픽 모델링 결과")
    print("=" * 60)

    # 토픽 정보 출력
    topic_info = topic_model.get_topic_info()
    print(f"\n발견된 토픽 수: {len(topic_info) - 1}")  # -1 제외
    print(f"아웃라이어 문서 수: {sum(1 for t in topics if t == -1)}")

    print("\n[토픽별 상세 정보]")
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        count = row['Count']

        if topic_id == -1:
            print(f"\n토픽 {topic_id} (아웃라이어): {count}개 문서")
        else:
            # 토픽 키워드 가져오기
            topic_words = topic_model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:5]]
            print(f"\n토픽 {topic_id}: {count}개 문서")
            print(f"   키워드: {', '.join(keywords)}")

    # 각 문서의 토픽 할당 결과
    print("\n[문서-토픽 할당 예시 (처음 10개)]")
    print("-" * 60)
    for i, (doc, topic) in enumerate(zip(documents[:10], topics[:10])):
        print(f"토픽 {topic:2d} | {doc[:40]}...")

    # 모델 저장
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'bertopic_model')
    topic_model.save(model_path, serialization='safetensors', save_ctfidf=True)
    print(f"\n모델 저장 완료: {model_path}")

    print("\n" + "=" * 60)
    print("BERTopic 기본 토픽 모델링 완료")
    print("=" * 60)

    return topic_model, topics, probs


if __name__ == "__main__":
    topic_model, topics, probs = main()
