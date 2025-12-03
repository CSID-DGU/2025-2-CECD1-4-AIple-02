import pandas as pd

def load_fewshot_data():
    mbti_groups = {
        'INF': [], 'INT': [], 'ISF': [], 'IST': [],
        'ENF': [], 'ENT': [], 'ESF': [], 'EST': []
    }

    try:
        df = pd.read_csv("mbti_script_dataset_converted.csv")
        df = df.dropna(axis=0)

        for idx, row in df.iterrows():
            mbti = str(row['I-E']).upper() + str(row['N-S']).upper() + str(row['T-F']).upper()

            if len(str(row['Output'])) < 75:
                if mbti in mbti_groups:
                    mbti_groups[mbti].append((row['Character'], row['Input'], row['Output']))

    except FileNotFoundError:
        print('No such File')

    keys = ['E', 'I', 'S', 'N', 'T', 'F']
    fewshot_dic = {k: {} for k in keys}

    for mbti, tuples in mbti_groups.items():
        for char in mbti:
            if mbti not in fewshot_dic[char]:
                fewshot_dic[char][mbti] = []
            fewshot_dic[char][mbti].extend(tuples)

    return fewshot_dic

fewshot_dic = load_fewshot_data()
