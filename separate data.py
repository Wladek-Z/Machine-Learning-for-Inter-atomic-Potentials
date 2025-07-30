folder1 = "" # filepath to .data file containing many structures
folder1 = "" # filepath to new directory which will contain the separated data

blocks = []

with open(folder1, "r") as file:
    current_block = []
    inside_block = False
    for line in file:
        if line.strip().startswith("begin"):
            current_block = []
            current_block.append(line)
        elif line.strip().startswith("end"):
            current_block.append(line)
            blocks.append(current_block)
        else:
            current_block.append(line)


for i, block in enumerate(blocks):
    with open(f"{folder2}/input-single-{i+1}.data", "w") as out_file:
        out_file.write("".join(block))



