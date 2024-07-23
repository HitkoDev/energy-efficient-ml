

if __name__ == "__main__":
    with open("all_new_features_hier_norm.csv") as f_old:
        with open("ostalaKoda/ilsvrc2012_features_50k.csv") as f_new:
            old_names = {}
            new_names = []
            names = []
            for i, line in enumerate(f_old):
                #if i == 0:
                #    continue
                parts = line.split(",")
                old_names[parts[0]] = parts
            for i, line in enumerate(f_new):
                if i == 0:
                    names = line.split(",")
                    to_remove = "perc_brightness_rms,avg_brightness,brightness_rms,edge_angle2,edge_angle3,edge_angle4,edge_angle5,edge_angle6,edge_angle7".split(",")
                    indices = [names.index(h) for h in to_remove]
                    in_indices = [x for x in range(len(names) - 1) if x not in indices]
                    in_indices.sort()
                    #continue
                try:
                    parts = line.split(",")
                    parts = [parts[x] for x in in_indices]
                    parts.insert(1, old_names[parts[0]][3])
                    parts.insert(1, old_names[parts[0]][2])
                    parts.insert(1, old_names[parts[0]][1])
                    new_names.append(parts)
                except:
                    pass
            print(new_names)

    with open("ostalaKoda/new_features.csv", "w") as f:
        for line in new_names:

            f.write(",".join(line))
            f.write("\n")