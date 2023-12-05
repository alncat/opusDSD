case $3 in
    kmeans)
        dsd eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/analyze.$2/kmeans$4/centers.txt -o $1/analyze.$2/kmeans$4/ --Apix $5 --encode-mode grad --pe-type vanilla --template-type conv
    ;;

    pc)
        dsd eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/analyze.$2/pc$4/z_pc.txt -o $1/analyze.$2/pc$4/ --Apix $5 --encode-mode grad --pe-type vanilla --template-type conv
    ;;
esac
