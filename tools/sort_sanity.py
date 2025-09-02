#!/usr/bin/env python3
"""
Sort Sanity Check Tool

Muestra los primeros N desempates en el sistema de ordenamiento total
para verificar el comportamiento determinístico.
"""

import sys
import argparse
import json
from typing import List, Dict, Any
from pathlib import Path

# Agregar el path del proyecto para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from egw_query_expansion.core.total_ordering import (
    OrderedRecord,
    sort_total,
    explain_tie_breaks,
    explain_comparison,
    serialize_ordered
)


def generate_synthetic_tie_cases(n_items: int = 20) -> List[OrderedRecord]:
    """Generar casos sintéticos con empates para testing"""
    items = []
    
    # Grupo 1: Empates perfectos en scores
    for i in range(n_items // 4):
        items.append(OrderedRecord(
            scores=(0.9, 0.1, 0.5),
            uids=(f"perfect_tie_{i:03d}", f"group_a", f"subcat_{i % 3}"),
            payload=f"perfect_{i}"
        ))
    
    # Grupo 2: Empates parciales (solo primeras dimensiones)
    for i in range(n_items // 4):
        items.append(OrderedRecord(
            scores=(0.8, 0.2, 0.1 + i * 0.1),  # Varía la tercera dimensión
            uids=(f"partial_tie_{i:03d}", f"group_b", f"variant_{i}"),
            payload=f"partial_{i}"
        ))
        
    # Grupo 3: Casos con valores especiales (inf, nan)
    items.append(OrderedRecord(
        scores=(float('inf'), 0.0),
        uids=("special_inf", "group_c"),
        payload="inf_case"
    ))
    
    items.append(OrderedRecord(
        scores=(float('nan'), 1.0),
        uids=("special_nan", "group_c"), 
        payload="nan_case"
    ))
    
    # Grupo 4: Casos normales sin empates
    for i in range(n_items - len(items)):
        items.append(OrderedRecord(
            scores=(0.7 - i * 0.01, 0.3 + i * 0.005),
            uids=(f"normal_{i:03d}", f"group_d"),
            payload=f"normal_{i}"
        ))
    
    return items


def analyze_tiebreaks(items: List[OrderedRecord], n_show: int = 10) -> Dict[str, Any]:
    """Analizar los desempates en el conjunto de items"""
    
    # Ordenar items
    ordered = sort_total(
        items,
        lambda r: r.scores,
        lambda r: r.uids,
        descending=True
    )
    
    # Obtener explicación de desempates
    tie_report = explain_tie_breaks(
        items,
        lambda r: r.scores, 
        lambda r: r.uids,
        descending=True
    )
    
    # Mostrar primeros N casos detallados
    detailed_comparisons = []
    for i in range(min(n_show - 1, len(ordered) - 1)):
        item_a = ordered[i]
        item_b = ordered[i + 1]
        
        comparison = explain_comparison(
            item_a.scores, item_a.uids,
            item_b.scores, item_b.uids,
            descending=True
        )
        
        detailed_comparisons.append({
            "rank_a": i,
            "rank_b": i + 1,
            "item_a": {
                "scores": item_a.scores,
                "uids": item_a.uids,
                "payload": item_a.payload
            },
            "item_b": {
                "scores": item_b.scores,
                "uids": item_b.uids,
                "payload": item_b.payload
            },
            "comparison": comparison
        })
    
    return {
        "total_items": len(items),
        "ordered_sample": [
            {
                "rank": i,
                "scores": item.scores,
                "uids": item.uids,
                "payload": item.payload
            }
            for i, item in enumerate(ordered[:n_show])
        ],
        "tie_analysis": tie_report,
        "detailed_comparisons": detailed_comparisons,
        "serialization": serialize_ordered(
            items[:n_show],
            lambda r: r.scores,
            lambda r: r.uids,
            descending=True
        )
    }


def main():
    parser = argparse.ArgumentParser(description="Sort Sanity Check - Analizar desempates determinísticos")
    parser.add_argument(
        "-n", "--num-items",
        type=int, 
        default=20,
        help="Número de items sintéticos a generar (default: 20)"
    )
    parser.add_argument(
        "-s", "--show",
        type=int,
        default=10, 
        help="Número de items a mostrar en detalle (default: 10)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Archivo JSON de salida (opcional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Generando {args.num_items} casos sintéticos con empates...")
    items = generate_synthetic_tie_cases(args.num_items)
    
    print(f"📊 Analizando desempates (mostrando primeros {args.show})...")
    analysis = analyze_tiebreaks(items, args.show)
    
    # Mostrar resumen en consola
    print("\n" + "="*60)
    print("RESUMEN DE ANÁLISIS DE DESEMPATES")
    print("="*60)
    
    print(f"Total de items: {analysis['total_items']}")
    print(f"Grupos de empate encontrados: {len(analysis['tie_analysis']['groups'])}")
    
    for i, group in enumerate(analysis['tie_analysis']['groups']):
        if group['size'] > 1:
            print(f"  Grupo {i+1}: {group['size']} items con empate")
            if args.verbose:
                print(f"    Posiciones UID usadas: {group['uid_positions_used_for_tiebreak']}")
    
    print(f"\nPrimeros {min(args.show, len(analysis['ordered_sample']))} items ordenados:")
    print("-" * 60)
    
    for item in analysis['ordered_sample']:
        scores_str = ", ".join(f"{s:.3f}" if isinstance(s, (int, float)) else str(s) for s in item['scores'])
        uids_str = ", ".join(item['uids'])
        print(f"#{item['rank']:2d}: scores=({scores_str}) | uids=({uids_str}) | {item['payload']}")
    
    if args.verbose and analysis['detailed_comparisons']:
        print(f"\nComparaciones detalladas (primeras {len(analysis['detailed_comparisons'])}):")
        print("-" * 60)
        
        for comp in analysis['detailed_comparisons'][:3]:  # Solo mostrar las primeras 3
            print(f"\nComparación #{comp['rank_a']} vs #{comp['rank_b']}:")
            print(f"  A: {comp['item_a']['payload']} | scores={comp['item_a']['scores']} | uids={comp['item_a']['uids']}")
            print(f"  B: {comp['item_b']['payload']} | scores={comp['item_b']['scores']} | uids={comp['item_b']['uids']}")
            print(f"  Resultado: {comp['comparison']['result']}")
            
            for step in comp['comparison']['steps']:
                if step['type'] == 'score':
                    print(f"    Score[{step['dimension']}]: {step['decision']}")
                elif step['type'] == 'uid':
                    print(f"    UID[{step['position']}]: {step['decision']}")
    
    # Guardar archivo si se especifica
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Análisis completo guardado en: {args.output}")
    
    print("\n✅ Análisis de desempates completado.")


if __name__ == "__main__":
    main()