"""Run all dataset converters in order."""
import convert_amod
import convert_annomi
import convert_empathetic
import convert_esconv
import convert_hope
import convert_psych8k

if __name__ == "__main__":
    for module in (convert_amod, convert_annomi, convert_esconv,
                   convert_psych8k, convert_empathetic, convert_hope):
        module.main()
