# zairachem fit -c 0.1 -d low -i 210922_plasmodium_NF54.csv
# zairachem fit -c 1 -d low -i 210922_mtb_H37Rv.csv
# zairachem fit -c 10 -d low -i 210922_cytotoxicity_CHO.csv
# zairachem fit -c 11.6 -d low -i 210922_clint_Human.csv
# zairachem fit -c 0.01 -d low -i 210922_enzymology_PvPI4KB.csv
# zairachem fit -c 0.1 -d low -i 210922_plasmodium_K1.csv
# zairachem fit -c 50 -d low -i 210922_cytotoxicity_HepG2.csv
zairachem fit -c 11.6 -d low -i 210922_clint_Mouse.csv
zairachem fit -c 11.6 -d low -i 210922_clint_Rat.csv
zairachem fit -i cyp_all_cyp3a4.csv
zairachem fit -d high -i 211117_solubility_74.csv
zairachem fit -i cyp_all_cyp2d6.csv
zairachem fit -i cyp_all_cyp2c9.csv
zairachem fit -i cyp_all_cyp2c19.csv
zairachem fit -d high -i chi2019.csv
zairachem fit -c 20 -d low -i 210922_mtb_H37Rv.csv -o 210922_mtb_H37Rv_20
