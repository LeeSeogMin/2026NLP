"""
8-5-topic-visualization.py
BERTopic 토픽 시각화

이 스크립트는 BERTopic의 다양한 시각화 기능을 보여준다:
1. Intertopic Distance Map (토픽 간 거리)
2. Topic Word Scores (토픽별 키워드 점수)
3. Topic Hierarchy (토픽 계층 구조)

실행 방법:
    python 8-5-topic-visualization.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

# 샘플 문서 데이터
documents = [
    # 경제
    "삼성전자 반도체 수출 사상 최대 실적 달성",
    "한국은행 기준금리 동결 결정 배경 분석",
    "코스피 지수 3000선 돌파 투자자 관심 급증",
    "현대차 전기차 생산량 전년 대비 30% 증가",
    "SK하이닉스 HBM 반도체 글로벌 시장 점유율",
    "원달러 환율 1300원대 진입 수출기업 영향",
    "카카오뱅크 대출 금리 인하 소비자 반응",
    "LG에너지솔루션 배터리 수주 역대 최고치",
    "반도체 업황 회복세 수출 증가 전망",
    "금리 인상 부동산 시장 영향 분석",

    # 스포츠
    "손흥민 시즌 15호골 기록 팀 승리 이끌어",
    "프로야구 개막전 관중 10만명 돌파 기대",
    "김연경 은퇴 후 배구 해설위원 활동 시작",
    "이강인 라리가 데뷔골 현지 언론 극찬",
    "한국 축구 대표팀 월드컵 예선 3연승 달성",
    "KBO리그 신인왕 경쟁 치열 팬들 주목",
    "LPGA 한국 선수 우승 국내 골프 열기",
    "축구 국가대표 평가전 승리 사기 충전",
    "야구 선수 FA 계약 역대 최고액 경신",
    "올림픽 메달 기대 종목 분석 전망",

    # 기술
    "챗GPT 활용한 업무 자동화 기업들 도입 확대",
    "메타버스 플랫폼 MAU 1억명 돌파 전망",
    "국내 AI 스타트업 투자 유치 1조원 시대",
    "자율주행차 레벨4 상용화 임박 규제 논의",
    "클라우드 컴퓨팅 시장 성장 디지털 전환 가속",
    "애플 비전프로 국내 출시 XR 시장 확대",
    "오픈AI GPT-5 개발 속도 경쟁 심화",
    "인공지능 윤리 가이드라인 정부 발표",
    "딥러닝 모델 학습 효율화 연구 성과",
    "로봇 기술 발전 제조업 혁신 가속화",

    # 문화
    "BTS 신곡 빌보드 1위 등극 K-POP 위상",
    "넷플릭스 한국 드라마 글로벌 인기 지속",
    "한국 영화 아카데미 후보 2년 연속 진출",
    "뮤지컬 시장 회복세 공연 예매율 상승",
    "K-POP 걸그룹 일본 오리콘 1위 석권",
    "한국 게임 글로벌 매출 순위 상위권 진입",
    "신인 아이돌 데뷔 음원차트 역주행 화제",
    "한류 콘텐츠 수출액 역대 최고 기록",
    "영화관 관객수 회복 극장가 활기",
    "드라마 OST 음원차트 상위권 장기 유지",
]


def create_topic_model():
    """BERTopic 모델 생성 및 학습"""
    print("BERTopic 모델 학습 중...")

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    umap_model = UMAP(
        n_neighbors=8,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=False
    )

    topics, probs = topic_model.fit_transform(documents)
    print(f"학습 완료. 발견된 토픽 수: {len(set(topics)) - (1 if -1 in topics else 0)}")

    return topic_model, topics, probs


def visualize_topic_keywords(topic_model, output_dir):
    """토픽별 키워드 막대 그래프 시각화"""
    print("\n[시각화 1] 토픽별 키워드 점수 그래프 생성 중...")

    topic_info = topic_model.get_topic_info()
    valid_topics = [t for t in topic_info['Topic'] if t != -1]

    if len(valid_topics) == 0:
        print("   시각화할 토픽이 없습니다.")
        return

    fig, axes = plt.subplots(1, min(len(valid_topics), 4), figsize=(14, 4))
    if len(valid_topics) == 1:
        axes = [axes]

    for idx, topic_id in enumerate(valid_topics[:4]):
        topic_words = topic_model.get_topic(topic_id)
        words = [word for word, _ in topic_words[:5]]
        scores = [score for _, score in topic_words[:5]]

        ax = axes[idx]
        y_pos = np.arange(len(words))
        ax.barh(y_pos, scores, color=f'C{idx}')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('c-TF-IDF Score')
        ax.set_title(f'Topic {topic_id}')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'topic_keywords.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {output_path}")


def visualize_topic_distribution(topic_model, topics, output_dir):
    """토픽 분포 시각화"""
    print("\n[시각화 2] 토픽 분포 그래프 생성 중...")

    topic_counts = {}
    for t in topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1

    # 토픽 ID와 개수
    sorted_topics = sorted(topic_counts.items())
    topic_ids = [str(t[0]) for t in sorted_topics]
    counts = [t[1] for t in sorted_topics]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['gray' if t == '-1' else f'C{i % 10}' for i, t in enumerate(topic_ids)]
    bars = ax.bar(topic_ids, counts, color=colors)

    ax.set_xlabel('Topic ID')
    ax.set_ylabel('Document Count')
    ax.set_title('Topic Distribution')

    # 막대 위에 개수 표시
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'topic_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {output_path}")


def visualize_document_topics(topic_model, documents, topics, output_dir):
    """문서-토픽 할당 히트맵"""
    print("\n[시각화 3] 문서-토픽 할당 표 생성 중...")

    # 토픽별로 문서 그룹화
    topic_docs = {}
    for doc, topic in zip(documents, topics):
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(doc)

    # 결과 출력
    print("\n   [토픽별 문서 할당 결과]")
    print("   " + "-" * 56)

    for topic_id in sorted(topic_docs.keys()):
        if topic_id == -1:
            topic_name = "Outliers"
        else:
            topic_words = topic_model.get_topic(topic_id)
            top_word = topic_words[0][0] if topic_words else "unknown"
            topic_name = f"Topic {topic_id} ({top_word})"

        print(f"\n   [{topic_name}] - {len(topic_docs[topic_id])}개 문서")
        for doc in topic_docs[topic_id][:3]:  # 각 토픽당 최대 3개만 출력
            print(f"      - {doc[:45]}...")


def print_topic_summary(topic_model, topics):
    """토픽 요약 정보 출력"""
    print("\n" + "=" * 60)
    print("토픽 모델링 결과 요약")
    print("=" * 60)

    topic_info = topic_model.get_topic_info()

    print(f"\n총 문서 수: {len(topics)}")
    print(f"발견된 토픽 수: {len(topic_info) - (1 if -1 in topic_info['Topic'].values else 0)}")
    print(f"아웃라이어 문서 수: {sum(1 for t in topics if t == -1)}")

    print("\n[토픽별 대표 키워드]")
    print("-" * 60)

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        count = row['Count']

        if topic_id == -1:
            print(f"Topic -1 (Outliers): {count}개 문서")
        else:
            topic_words = topic_model.get_topic(topic_id)
            keywords = ', '.join([w for w, _ in topic_words[:5]])
            print(f"Topic {topic_id} ({count}개): {keywords}")


def main():
    """메인 함수: 토픽 모델링 및 시각화"""

    print("=" * 60)
    print("BERTopic 토픽 시각화")
    print("=" * 60)
    print(f"\n총 문서 수: {len(documents)}")

    # 출력 디렉토리 설정
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 모델 생성 및 학습
    topic_model, topics, probs = create_topic_model()

    # 토픽 요약 출력
    print_topic_summary(topic_model, topics)

    # 시각화 생성
    visualize_topic_keywords(topic_model, output_dir)
    visualize_topic_distribution(topic_model, topics, output_dir)
    visualize_document_topics(topic_model, documents, topics, output_dir)

    # Plotly 기반 인터랙티브 시각화 (HTML 저장)
    print("\n[시각화 4] Plotly 인터랙티브 시각화 생성 중...")

    try:
        # 토픽 간 거리 시각화
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(os.path.join(output_dir, 'intertopic_distance.html'))
        print(f"   저장: {os.path.join(output_dir, 'intertopic_distance.html')}")

        # 막대 그래프 시각화
        fig_barchart = topic_model.visualize_barchart()
        fig_barchart.write_html(os.path.join(output_dir, 'topic_barchart.html'))
        print(f"   저장: {os.path.join(output_dir, 'topic_barchart.html')}")
    except Exception as e:
        print(f"   Plotly 시각화 생성 중 오류: {e}")

    print("\n" + "=" * 60)
    print("시각화 완료")
    print("=" * 60)

    return topic_model, topics


if __name__ == "__main__":
    topic_model, topics = main()
