"""
8-6-dynamic-topics.py
Dynamic Topic Modeling - 시간에 따른 토픽 변화 분석

이 스크립트는 BERTopic의 Dynamic Topic Modeling 기능을 사용하여
시간에 따른 토픽의 변화를 추적하고 시각화한다.

실행 방법:
    python 8-6-dynamic-topics.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import random
from collections import defaultdict

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

# 시간대별 샘플 문서 데이터 (AI 관련 뉴스 시뮬레이션)
# 2023년부터 2025년까지의 AI 관련 뉴스 제목

documents_with_timestamps = [
    # 2023년 초 - ChatGPT 등장 초기
    ("ChatGPT 출시 3개월 사용자 1억명 돌파 기록", "2023-02-01"),
    ("오픈AI 챗GPT 기업용 버전 출시 예고", "2023-02-15"),
    ("구글 바드 AI 챗봇 공개 경쟁 본격화", "2023-02-20"),
    ("챗GPT 교육계 논란 대학 과제 부정행위 우려", "2023-03-01"),
    ("마이크로소프트 빙 AI 검색 통합 발표", "2023-03-10"),
    ("AI 챗봇 활용 기업 업무 효율 30% 향상", "2023-03-20"),

    # 2023년 중반 - GPT-4 출시
    ("GPT-4 출시 멀티모달 AI 시대 개막", "2023-04-01"),
    ("GPT-4 의사 면허 시험 합격 수준 성능 입증", "2023-04-15"),
    ("AI 이미지 생성 저작권 분쟁 법적 논의 활발", "2023-05-01"),
    ("오픈AI 기업가치 300억달러 평가 투자 유치", "2023-05-20"),
    ("EU AI 규제법 합의 세계 최초 포괄적 법안", "2023-06-01"),
    ("메타 오픈소스 LLM 라마2 공개 업계 파장", "2023-06-15"),

    # 2023년 후반 - AI 확산
    ("생성형 AI 도입 기업 전년 대비 3배 증가", "2023-08-01"),
    ("AI 반도체 수요 급증 엔비디아 실적 호조", "2023-08-20"),
    ("국내 AI 스타트업 투자 유치 1조원 돌파", "2023-09-01"),
    ("AI 윤리 가이드라인 정부 발표 규제 논의", "2023-09-15"),
    ("오픈AI 내부 갈등 샘 알트만 CEO 해임 파문", "2023-11-20"),
    ("샘 알트만 오픈AI 복귀 이사회 재구성", "2023-11-25"),

    # 2024년 초 - AI 에이전트 시대
    ("AI 에이전트 자율적 업무 수행 기술 발전", "2024-01-15"),
    ("소라 AI 동영상 생성 충격적 품질 화제", "2024-02-01"),
    ("구글 제미나이 울트라 GPT-4 능가 주장", "2024-02-10"),
    ("클로드3 출시 벤치마크 최고 성능 기록", "2024-03-01"),
    ("AI PC 시대 개막 온디바이스 AI 주목", "2024-03-15"),
    ("애플 AI 전략 WWDC 발표 기대감 상승", "2024-04-01"),

    # 2024년 중반 - 멀티모달 AI
    ("GPT-4o 출시 음성 실시간 대화 가능", "2024-05-15"),
    ("애플 인텔리전스 공개 온디바이스 AI", "2024-06-10"),
    ("AI 검색 시장 경쟁 구글 점유율 위협", "2024-07-01"),
    ("오픈소스 LLM 성능 향상 라마3 공개", "2024-07-20"),
    ("AI 코딩 어시스턴트 개발자 70% 활용", "2024-08-01"),
    ("AI 저작권 소송 뉴욕타임스 오픈AI 피소", "2024-08-15"),

    # 2024년 후반 - AI 추론 모델
    ("오픈AI o1 추론 모델 공개 사고력 향상", "2024-09-15"),
    ("AI 에이전트 상용화 본격 시작 전망", "2024-10-01"),
    ("AI 반도체 미중 갈등 수출 규제 강화", "2024-10-20"),
    ("구글 제미나이2 발표 AI 경쟁 가열", "2024-12-01"),
    ("2024년 AI 투자 사상 최대 기록 갱신", "2024-12-15"),
    ("AI 일자리 영향 보고서 발표 우려와 기대", "2024-12-20"),

    # 2025년 초 - AI 에이전트 상용화
    ("AI 에이전트 엔터프라이즈 도입 가속", "2025-01-10"),
    ("자율 AI 시스템 안전성 논의 활발", "2025-01-20"),
    ("AGI 달성 시기 전문가 전망 엇갈려", "2025-02-01"),
    ("AI 규제 글로벌 표준화 논의 본격화", "2025-02-15"),
    ("AI 비용 절감으로 중소기업 도입 확대", "2025-03-01"),
    ("AI 의료 진단 정확도 전문의 수준 도달", "2025-03-15"),
]


def parse_timestamps(docs_with_ts):
    """문서와 타임스탬프 분리"""
    documents = [doc for doc, _ in docs_with_ts]
    timestamps = [datetime.strptime(ts, "%Y-%m-%d") for _, ts in docs_with_ts]
    return documents, timestamps


def create_topic_model(documents):
    """BERTopic 모델 생성 및 학습"""
    print("\nBERTopic 모델 학습 중...")

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    umap_model = UMAP(
        n_neighbors=5,
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
        top_n_words=8,
        verbose=False
    )

    topics, probs = topic_model.fit_transform(documents)
    print(f"학습 완료. 발견된 토픽 수: {len(set(topics)) - (1 if -1 in topics else 0)}")

    return topic_model, topics, probs


def analyze_topics_over_time(topic_model, documents, timestamps, topics):
    """시간에 따른 토픽 변화 분석"""
    print("\n시간별 토픽 분석 중...")

    # topics_over_time 계산
    topics_over_time = topic_model.topics_over_time(
        documents,
        timestamps,
        nr_bins=6  # 6개 기간으로 분할
    )

    return topics_over_time


def visualize_topics_over_time(topic_model, topics_over_time, output_dir):
    """시간별 토픽 트렌드 시각화"""
    print("\n[시각화] 시간별 토픽 트렌드 그래프 생성 중...")

    # Matplotlib으로 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    # 토픽별로 시간에 따른 빈도 집계
    topic_ids = topics_over_time['Topic'].unique()
    topic_ids = [t for t in topic_ids if t != -1]  # 아웃라이어 제외

    for topic_id in topic_ids[:5]:  # 상위 5개 토픽만 시각화
        topic_data = topics_over_time[topics_over_time['Topic'] == topic_id]
        timestamps = topic_data['Timestamp']
        frequencies = topic_data['Frequency']

        # 토픽 레이블 가져오기
        topic_words = topic_model.get_topic(topic_id)
        label = topic_words[0][0] if topic_words else f"Topic {topic_id}"

        ax.plot(timestamps, frequencies, marker='o', label=f"Topic {topic_id}: {label}")

    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Topic Trends Over Time')
    ax.legend(loc='upper left', fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'topics_over_time.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {output_path}")

    # Plotly 인터랙티브 시각화
    try:
        fig_plotly = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=5
        )
        plotly_path = os.path.join(output_dir, 'topics_over_time.html')
        fig_plotly.write_html(plotly_path)
        print(f"   저장: {plotly_path}")
    except Exception as e:
        print(f"   Plotly 시각화 오류: {e}")


def print_time_analysis(topic_model, topics_over_time):
    """시간별 토픽 분석 결과 출력"""
    print("\n" + "=" * 60)
    print("시간별 토픽 변화 분석 결과")
    print("=" * 60)

    # 기간별로 그룹화
    periods = topics_over_time.groupby('Timestamp')

    for timestamp, group in list(periods)[:4]:  # 처음 4개 기간만 출력
        print(f"\n[{timestamp.strftime('%Y-%m')}]")
        print("-" * 40)

        # 해당 기간의 토픽별 빈도
        for _, row in group.iterrows():
            topic_id = row['Topic']
            freq = row['Frequency']

            if topic_id == -1:
                continue

            words = row['Words'].split(', ')[:3] if 'Words' in row else []
            if not words:
                topic_words = topic_model.get_topic(topic_id)
                words = [w for w, _ in topic_words[:3]]

            print(f"   Topic {topic_id}: {', '.join(words)} (freq: {freq})")


def print_topic_evolution(topic_model, topics_over_time):
    """토픽 진화 패턴 분석"""
    print("\n" + "=" * 60)
    print("토픽 진화 패턴 분석")
    print("=" * 60)

    # 각 토픽의 시간별 키워드 변화 분석
    topic_ids = [t for t in topics_over_time['Topic'].unique() if t != -1]

    for topic_id in topic_ids[:3]:  # 상위 3개 토픽
        topic_data = topics_over_time[topics_over_time['Topic'] == topic_id]

        if len(topic_data) == 0:
            continue

        # 토픽 기본 정보
        topic_words = topic_model.get_topic(topic_id)
        main_keywords = [w for w, _ in topic_words[:5]]

        print(f"\n[Topic {topic_id}]")
        print(f"   핵심 키워드: {', '.join(main_keywords)}")

        # 시간별 빈도 변화
        first_freq = topic_data['Frequency'].iloc[0] if len(topic_data) > 0 else 0
        last_freq = topic_data['Frequency'].iloc[-1] if len(topic_data) > 0 else 0

        if first_freq > 0:
            change_rate = ((last_freq - first_freq) / first_freq) * 100
            trend = "상승" if change_rate > 0 else "하락" if change_rate < 0 else "유지"
            print(f"   빈도 변화: {first_freq:.0f} -> {last_freq:.0f} ({trend}, {abs(change_rate):.1f}%)")


def main():
    """메인 함수: Dynamic Topic Modeling 실행"""

    print("=" * 60)
    print("Dynamic Topic Modeling - 시간별 토픽 변화 분석")
    print("=" * 60)
    print(f"\n분석 기간: 2023-02 ~ 2025-03")
    print(f"총 문서 수: {len(documents_with_timestamps)}")

    # 출력 디렉토리 설정
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 문서와 타임스탬프 분리
    documents, timestamps = parse_timestamps(documents_with_timestamps)

    # 모델 생성 및 학습
    topic_model, topics, probs = create_topic_model(documents)

    # 토픽 정보 출력
    topic_info = topic_model.get_topic_info()
    print("\n[토픽 개요]")
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        count = row['Count']
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            keywords = ', '.join([w for w, _ in topic_words[:4]])
            print(f"   Topic {topic_id} ({count}개): {keywords}")

    # 시간별 토픽 분석
    topics_over_time = analyze_topics_over_time(topic_model, documents, timestamps, topics)

    # 시간별 분석 결과 출력
    print_time_analysis(topic_model, topics_over_time)

    # 토픽 진화 패턴 분석
    print_topic_evolution(topic_model, topics_over_time)

    # 시각화 생성
    visualize_topics_over_time(topic_model, topics_over_time, output_dir)

    print("\n" + "=" * 60)
    print("Dynamic Topic Modeling 분석 완료")
    print("=" * 60)

    return topic_model, topics_over_time


if __name__ == "__main__":
    topic_model, topics_over_time = main()
