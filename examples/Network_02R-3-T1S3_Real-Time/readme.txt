Files used by more than one scenario of this dataset are kept in the root folder. 
Files unique to a scenario are kept in the appropriat scenario_n folder.
The file inputfiles.ini is a text file that indicates where each of the four file types may be found.
The beginning of the file is marked with [INPUTS].
A period (.) after the file type indicates the file is in the root folder.
An x after the file type indicates the file is in the scenario folder(s).
RAW files are, by definition, unique for each scenario
CON files typically vary somewhat for each scenario but are sometimes the same for all scenarios.
ROP files typically are the same for all scenarios, as are the INL files.
The inputfiles.ini file avoids redundant copies of files.

This is a GO Competition Challenge 1 Network_02 dataset with the following characteristics.
			  buses	500
			  loads	281
		   fixed_shunts	  5
		     generators	224
        nontransformer_branches	540
		   transformers	193
    			  areas	  1
	        switched_shunts	 39
 	  generator inl records	224
     generator dispatch records	224
  active power dispatch records	224
piecewise linear cost functions	224
                      scenarios   3
              Low(1) Medium(2) High(3)
Network_02R-060	48	30	11
