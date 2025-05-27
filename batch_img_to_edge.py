import subprocess

datasets = ["Rain100L","Rain12", "Rain800", "DDN-SIRR-syn", "DDN-SIRR-real", "GT-rain"]

for dataset in datasets:
    print(f"🚀 開始處理：{dataset}")
    command = f"python img_to_edge.py --dataset={dataset}"
    result = subprocess.run(command, shell=True)

    if result.returncode == 0:
        print(f"✅ {dataset} 完成！\n")
    else:
        print(f"❌ {dataset} 發生錯誤，Return Code: {result.returncode}\n")

print("🎉 全部資料集處理完畢！")