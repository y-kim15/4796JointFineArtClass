import pandas
import unidecode

special_char = { "aEURoe": "", "aEUR": "", "a%0" : "e", "EUR(tm)" : "", "a3" : "o", "a(c)": "e", "aa": "a", "aSS": "c"}
class ProcessCSV:
    def read_csv(path):
        # read in csv
        data = pandas.read_csv(path, encoding='utf-8-sig')
        headers = list(data[:0])
        data["agent_display"] = data["agent_display"].apply(lambda x: x.lower())
        data["title_display"] = data["title_display"].apply(lambda x: x.lower())
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
            if rm and str in old:
                print("Yes")
                new = old.replace(str, "")
            else:
                old = unidecode.unidecode(old)

                new_str = old
                for j in special_char:
                    if j in old:
                        new_str = new_str.replace(j, special_char[j])
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

        return lcs_set

    def

# df.loc[df.index[#], 'NAME']

if __name__ == '__main__':
    path = "./cleaning_small.csv"
    data, headers = ProcessCSV.read_csv(path)
    data = ProcessCSV.remove_str(data, "agent_display", True, str = "?")
    data = ProcessCSV.remove_str(data, "agent_display", True, str = ".")
    data = ProcessCSV.remove_str(data, "agent_display", True, str = ";")
    data = ProcessCSV.remove_str(data, "agent_display", True, str = "\'")
    data = ProcessCSV.remove_str(data, "title_display", True, str = "?")
    data = ProcessCSV.remove_str(data, "title_display", True, str = ".")
    data = ProcessCSV.remove_str(data, "title_display", True, str = ";")
    data = ProcessCSV.remove_str(data, "title_display", True, str = "\'")
    data = ProcessCSV.remove_str(data, "agent_display", False)
    data = ProcessCSV.remove_str(data, "title_display", False)
    #data = ProcessCSV.assign_id(data)
    data.to_csv("./result.csv", header=headers, index=False)
    print(headers)

    data = data.as_matrix()
    print(data)
