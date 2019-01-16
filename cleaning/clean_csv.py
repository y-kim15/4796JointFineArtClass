import pandas
import unidecode
import os
import re
import sys
from PIL import Image
import requests
import time

# methods to process idb csv file
special_char = {"aEURoe": "", "aEUR": "", "a%0": "e", "EUR(tm)": "", "a3": "o", "a(c)": "e", "aa": "a", "aSS": "c"}


class Clean:
    @staticmethod
    def read_csv(path):
        # read in csv
        data = pandas.read_csv(path, encoding='utf-8-sig')
        headers = list(data[:0])

        data["agent_display"] = data["agent_display"].astype(str).apply(lambda x: x.lower())
        data["title_display"] = data["title_display"].astype(str).apply(lambda x: x.lower())
        data["date_display"] = data["date_display"].apply(lambda x: str(x).lower())
        return data, headers

    @staticmethod
    def remove_str(data, col_name, rm, **string):
        # remove a character from cells of a certain column
        # if true remove provided, else go through check on all special chars
        new_data = pandas.DataFrame.copy(data)
        i = 0
        if rm:
            str = string["str"]
        for (new, old) in zip(new_data[col_name], data[col_name]):
            if rm:
                if type(str) is list:
                    new = old
                    for char in str:
                        if char in old:
                            new = new.replace(char, "")
                else:
                    if str in old:
                        new = old.replace(str, "")
                # print("Yes")
                # new = old.replace(str, "")
            else:
                old = unidecode.unidecode(old)

                new_str = old
                for j in special_char:
                    if j in old:
                        new_str = new_str.replace(j, special_char[j])
                # if new_str == "":
                #    new_str = old
                new = new_str

            new_data.iloc[i][col_name] = new
            i += 1
        return new_data

    @staticmethod
    def assign_id(df, col_name):
        # creates unique id for every row in format of xx-yy where xx for artist
        # yy for work count of an artist
        # col_name to generate the code on
        new_data = pandas.DataFrame.copy(df)  # MIGHT NOT NEED THIS
        i = 0  # total
        current = ""  # current artist
        current_i = 0  # current artist count
        current_j = 0  # current artist work count
        # for(new, old) in zip (new_data["agent_display"], data["agent_display"]):
        artist_count = {}
        work_count = {}
        temp_i = -1
        for old in df[col_name]:
            if old == current:
                current_j += 1
            else:
                temp_i = current_i
                if old in artist_count:
                    current_i = artist_count[old]
                    current_j = work_count[current_i] + 1
                else:
                    work_count[current_i] = current_j
                    current_i = temp_i + 1
                    current_j = 1
                    artist_count[old] = current_i
                current = old
            work_count[current_i] = current_j
            new = str(current_i) + "-" + str(current_j)
            new_data.id_agent[i] = current_i
            new_data.id_count[i] = current_j
            #new_data.iloc[i, "id_agent"] = current_i
            #new_data.iloc[i, "id_count"] = current_j
            i += 1

        new_data.id_agent = new_data.id_agent.astype(int)
        new_data.id_count = new_data.id_count.astype(int)
        return new_data

    @staticmethod
    # https://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php
    def lcs(S, T):
        m = len(S)
        n = len(T)
        counter = [[0] * (n + 1) for x in range(m + 1)]
        longest = 0
        lcs_set = set()
        for i in range(m):
            for j in range(n):
                if S[i] == T[j]:
                    c = counter[i][j] + 1
                    counter[i + 1][j + 1] = c
                    if c > longest:
                        lcs_set = set()
                        longest = c
                        lcs_set.add(S[i - c + 1:i + 1])
                    elif c == longest:
                        lcs_set.add(S[i - c + 1:i + 1])
        lcs_list = list(lcs_set)
        if len(lcs_list) == 1:
            return lcs_list[0]
        else:
            return ""

    @staticmethod
    def filter_to_file(path, data, str):
        # drop rows to be opted from the data and writes to a separate file
        new_data = pandas.DataFrame.copy(data)
        headers = list(data[:0])
        dropped = pandas.DataFrame(columns=headers)

        for index, row in data.iterrows():
            if type(str) is list:
                for w in str:
                    if w in row["agent_display"]:
                        dropped = dropped.append(data.iloc[index].copy(deep=True))
                        new_data = new_data.drop(data.index[index])
                        break
            else:
                if str in row["agent_display"]:
                    dropped = dropped.append(data.iloc[index].copy(deep=True))
                    new_data = new_data.drop(data.index[index])

            if row["agent_display"] == "anon":
                dropped = dropped.append(data.iloc[index].copy(deep=True))
                new_data = new_data.drop(data.index[index])

        dropped.to_csv(path, header=headers, index=False)
        return new_data

    @staticmethod
    def match_names(data):
        new_data = pandas.DataFrame.copy(data)
        first = ""
        i = 0
        for index, row in new_data.iterrows():
            # print(type(row["agent_display"]))
            # print("current name is %s", row["agent_display"])
            # print("current first is %s", first)
            if first == "":
                first = row["agent_display"]
            else:
                out = Clean.lcs(first, row["agent_display"])
                # print("out is %s", out)
                if Clean.lcs(first, row["agent_display"]) == first:
                    # print("YES")
                    new_data.iloc[i]["agent_display"] = first
                else:
                    first = row["agent_display"]
            i += 1
        return new_data

    @staticmethod
    def find_artist_list(min, path):
        # find minimum number of works artists should have
        df = pandas.read_csv(path)
        headers = list(df[:0])
        i = 0
        rows = df.loc[df["id"].str.contains("-" + str(min))]

        for index, row in rows.iterrows():
            # print(row["agent_display"])
            i += 1
        print("total number of artist is", i)

    @staticmethod
    def check_if_similar(sample_name, other_name, type):
        # method to compare how similar the names are between the two, t artist name, f title
        # applicable for checking equivalence of artist and title names
        # sample_name: name of file, other_name: name on csv
        sample = sample_name.split("-")
        other = other_name.split("-")
        if type:
            # compare surnames
            if sample[-1] == other[0]:
                if len(sample) > 1 and len(other) > 1:
                    if sample[0][0] == other[1][0]:
                        return True
                else:
                    return True
        elif type is None:
            # date
            if sample_name in other_name:
                return True
        else:
            if sample_name == other_name or sample_name in other_name:
                return True
        return False

    @staticmethod
    def check_if_true(df, dropped_df, sub_df, artist_str, title_str, d, **date):
        if sub_df is not None and Clean.check_if_similar(sub_df[0]["agent_display"], artist_str, True):
            artist_rows = sub_df
        else:
            artist_rows = df.loc[df["agent_display"].str.contains(artist_str)]
        total_drop = 0
        new_df = pandas.DataFrame.copy(df)
        for i in range(artist_rows.shape[0]):
            if Clean.check_if_similar(title_str, artist_rows.iloc[i]["title_display"], False):
                dropped_df = dropped_df.append(artist_rows.iloc[i])
                new_df.drop(artist_rows.index[i], inplace=True)
                print("Drop row with artist: ", artist_str, " and title: ", artist_rows.iloc[i]["title_display"])
                total_drop += 1

        if d:
            date_str = date["date"]
            # date = df["date_display"].apply(lambda x: Clean.check_if_similar(date_str, x, None))

        return new_df, dropped_df, sub_df, total_drop

    @staticmethod
    def remove_match_images(data_path, input_csv, output_csv, dropped_csv):
        # read in csv
        df = pandas.read_csv(input_csv, encoding='utf-8-sig')
        df.set_index(['id'])
        headers = list(df[:0])
        dirs = os.listdir(data_path)
        date = ""
        sub_df = None
        dropped_df = pandas.DataFrame(columns=headers)
        total_drop = 0
        for dir in dirs:
            styles = os.listdir(os.path.join(data_path, dir))
            print("=========Dir type ", dir)
            for style in styles:
                style_drop = 0
                works = os.listdir(os.path.join(data_path, dir, style))
                for work in works:
                    strs = work.split("_")
                    artist = strs[0]
                    title = strs[1].replace('.jpg', '')
                    split = title.rsplit("-", 1)
                    if len(split) > 1:
                        # if for case xxxx.jpg
                        if re.match('^\d{4}$', split[1]):
                            date = split[1]
                            title = split[0]
                        # elif for case where there is xxxx-x.jpg format where year is former
                        elif len(split[0].rsplit("-",1)) > 1 and re.match('^\d{4}$', split[0].rsplit("-", 1)[1]):

                            sub = split[0].rsplit("-", 1)
                            date = sub[1]
                            title = sub[0]
                    if date != "":
                        df, dropped_df, sub_df, drop = Clean.check_if_true(df, dropped_df, sub_df, artist, title, True, date=date)
                    else:
                        df, dropped_df, sub_df, drop = Clean.check_if_true(df, dropped_df, sub_df, artist, title, False)
                    style_drop += drop
                print("Style " + style + " dropped: ", str(style_drop))
                total_drop += style_drop
            print("Total for dir type ", dir,  " is ", str(total_drop))
        print("Total dropped is ", str(total_drop))
        df.to_csv(output_csv, header=headers, index=False)
        dropped_df.to_csv(dropped_csv, header=headers, index=False)
        return df

    @staticmethod
    def _download_image(file_name, url):
        try:
            img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
            img.save(file_name + '.jpg', 'JPEG')
        except OSError:
            print('Error downloading image', file_name)

    @staticmethod
    def download_image_database(input_csv, target_path):
        # method referred from rasta.python.utils.load_data.download_wikipaintings() and _download_image()
        df = pandas.read_csv(input_csv, encoding='utf-8-sig')
        df.set_index(['id'])
        n = 0
        for index, row in df.iterrows():
            filename = target_path + '/' + row['agent_display'] + "_" + row['title_display']
            Clean._download_image(filename, row['full size'])
            n += 1
            if n % 5 == 0:
                sys.stdout.flush()
                time.sleep(1)
        return df



if __name__ == '__main__':
    """path = "./data/cleaning_1710.csv"
    data, headers = Clean.read_csv(path)
    data = Clean.remove_str(data, "agent_display", False)
    data = Clean.remove_str(data, "title_display", False)
    data = Clean.remove_str(data, "agent_display", True, str = ["?", ":", ";", ".", "\'", "\"", "~", "+"])
    data = Clean.remove_str(data, "title_display", True, str = ["?", ":", ";", ".", "\'", "\"", "~"])
    data = Clean.remove_str(data, "date_display", True, str = "?")
    data = Clean.filter_to_file("./data/multi_unknown_author_large.csv", data, ["&", "-and-", "-or-", "after","-with-", "anon-", "anonymous","wkshp", "wkshop", "workshop"])
    data = Clean.remove_str(data, "agent_display", True, str = ["-attr", "-attrib", "-sch", "-sch", "-school", "-school-of"])
    data = Clean.match_names(data)
    #data.to_csv("./data/result_for_style_with_wkp.csv", header=headers, index=False)
    #data = Clean.remove_str(data, "agent_display", True, str = [])
    data = Clean.assign_id(data, "agent_display")
    data.to_csv("./data/result_large.csv", header=headers, index=False)
    #print(headers)"""
    #Clean.find_artist_list(200, "../data/result_large.csv")  # 50 - 134 artists, #100 - 54 artists #150 - 28 artists #200  - 18 artist
    #Clean.remove_match_images("../data/wikipaintings_full", "../data/result_large.csv",
     #                         "../data/filtered_full_large.csv", "../data/dropped_full_large.csv" )
    data = pandas.read_csv("../data/filtered_full_large.csv", header=0)
    headers = list(data[:0])
    data = Clean.assign_id(data, "agent_display")
    data.to_csv("../data/filtered_full_large1.csv", header=headers, index=False)


