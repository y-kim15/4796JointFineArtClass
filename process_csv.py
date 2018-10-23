import pandas
import unidecode
import os
# methods to process idb csv file
special_char = { "aEURoe": "", "aEUR": "", "a%0" : "e", "EUR(tm)" : "", "a3" : "o", "a(c)": "e", "aa": "a", "aSS": "c"}

class ProcessCSV:
    def read_csv(path):
        # read in csv
        data = pandas.read_csv(path, encoding='utf-8-sig')
        headers = list(data[:0])

        data["agent_display"] = data["agent_display"].astype(str).apply(lambda x: x.lower())
        data["title_display"] = data["title_display"].astype(str).apply(lambda x: x.lower())
        data["date_display"] = data["date_display"].apply(lambda x: str(x).lower())
        return data, headers

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
                #print("Yes")
                #new = old.replace(str, "")
            else:
                old = unidecode.unidecode(old)

                new_str = old
                for j in special_char:
                    if j in old:
                        new_str = new_str.replace(j, special_char[j])
                for j in special_char2:
                    if j in old:
                        new_str = new_str.replace(j, special_char2[j])
                #if new_str == "":
                #    new_str = old
                new = new_str

            new_data.iloc[i][col_name] = new
            i += 1
        return new_data

    def assign_id(data):
        # creates unique id for every row in format of xx-yy where xx for artist
        # yy for work count of an artist
        new_data = pandas.DataFrame.copy(data) # MIGHT NOT NEED THIS
        i = 0  # total
        current = "" # current artist
        current_i = 0 # current artist count
        current_j = 0 # current artist work count
        #for(new, old) in zip (new_data["agent_display"], data["agent_display"]):
        for old in data["agent_display"]:
            if i == 0:
                current_i += 1
                current_j += 1
                current = old
            else:
                if old == current:
                     current_j += 1
                else:
                    current_i += 1
                    current_j = 1
                    current = old
            new = str(current_i)+"-"+str(current_j)
            data.iloc[i]["id"] = new
            i += 1
        return data

    # https://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php
    def lcs(S,T):
        m = len(S)
        n = len(T)
        counter = [[0]*(n+1) for x in range(m+1)]
        longest = 0
        lcs_set = set()
        for i in range(m):
            for j in range(n):
                if S[i] == T[j]:
                    c = counter[i][j] + 1
                    counter[i+1][j+1] = c
                    if c > longest:
                        lcs_set = set()
                        longest = c
                        lcs_set.add(S[i-c+1:i+1])
                    elif c == longest:
                        lcs_set.add(S[i-c+1:i+1])
        lcs_list = list(lcs_set)
        if len(lcs_list) == 1 :
            return lcs_list[0]
        else:
            return ""

    def filter_to_file(path, data, str):
        # drop rows to be opted from the data and writes to a separate file
        new_data = pandas.DataFrame.copy(data)
        headers = list(data[:0])
        dropped = pandas.DataFrame(columns = headers)

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

    def match_names(data):
        new_data = pandas.DataFrame.copy(data)
        first = ""
        i = 0
        for index, row in new_data.iterrows():
            #print(type(row["agent_display"]))
            #print("current name is %s", row["agent_display"])
            #print("current first is %s", first)
            if first == "":
                first = row["agent_display"]
            else:
                out = ProcessCSV.lcs(first, row["agent_display"])
                #print("out is %s", out)
                if ProcessCSV.lcs(first, row["agent_display"]) == first:
                    #print("YES")
                    new_data.iloc[i]["agent_display"] = first
                else:
                    first = row["agent_display"]
            i += 1
        return new_data



if __name__ == '__main__':
    path = "./data/cleaning_1710.csv"
    data, headers = ProcessCSV.read_csv(path)
    data = ProcessCSV.remove_str(data, "agent_display", False)
    data = ProcessCSV.remove_str(data, "title_display", False)
    data = ProcessCSV.remove_str(data, "agent_display", True, str = ["?", ":", ";", ".", "\'", "\"", "~", "+"])
    data = ProcessCSV.remove_str(data, "title_display", True, str = ["?", ":", ";", ".", "\'", "\"", "~"])
    data = ProcessCSV.remove_str(data, "date_display", True, str = "?")
    data = ProcessCSV.filter_to_file("./data/multi_unknown_author_large.csv", data, ["&", "-and-", "-or-", "after","-with-", "anon-", "anonymous","wkshp", "wkshop", "workshop"])
    data = ProcessCSV.remove_str(data, "agent_display", True, str = ["-attr", "-attrib", "-sch", "-sch", "-school", "-school-of"])
    data = ProcessCSV.match_names(data)
    #data.to_csv("./data/result_for_style_with_wkp.csv", header=headers, index=False)
    #data = ProcessCSV.remove_str(data, "agent_display", True, str = [])
    data = ProcessCSV.assign_id(data)
    data.to_csv("./data/result_large.csv", header=headers, index=False)
    #print(headers)

    data = data.as_matrix()
    #print(data)
