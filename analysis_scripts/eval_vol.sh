case $3 in
    kmeans)
        dsd eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/analyze.$2/kmeans$4/centers.txt -o $1/analyze.$2/kmeans$4/ --Apix $5 --num-bodies $6
    ;;

    pc)
        dsd eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/analyze.$2/pc$4/z_pc.txt -o $1/analyze.$2/pc$4/ --Apix $5 --num-bodies $6
    ;;

    dpc)
        dsd eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/defanalyze.$2/pc$4/z_pc.txt -o $1/defanalyze.$2/pc$4/ --Apix $5 --deform --masks $6 --template-z $1/analyze.$2/kmeans$7/centers.txt --template-z-ind $8
    ;;
esac
