from flask import Blueprint, request, jsonify
from models import Procedure
from extensions import db
from collections import defaultdict

procedure_detail_bp = Blueprint("procedure_detail", __name__)

@procedure_detail_bp.route("/procedure-detail", methods=["GET"])
def procedure_detail():
    procedure_id = request.args.get("procedure_id")
    if not procedure_id:
        return jsonify({"error": "Missing procedure_id"}), 400

    # Fetch procedure by id (primary key)
    procedure = Procedure.query.get(int(procedure_id))
    if not procedure:
        return jsonify({"error": "Procedure not found"}), 404

    # Get all surgeons associated with this procedure
    surgeons = [
        {
            'id': ps.surgeon.id,
            'name': ps.surgeon.name
        }
        for ps in procedure.procedure_surgeons
    ]

    # Group materials by type -> subtype -> materials
    grouped_materials = defaultdict(lambda: defaultdict(list))
    for pm in procedure.procedure_materials:
        material = pm.material
        grouped_materials[material.type][material.subtype].append({
            'id': material.id,
            'name': material.name,
            'original_price': material.original_price,
            'optimized_price': material.optimized_price,
            'surgeon_specific_action': pm.surgeon_specific_action
        })

    # Convert defaultdict to regular dict for JSON serialization
    materials_grouped = {
        m_type: {
            m_subtype: mat_list
            for m_subtype, mat_list in subtypes.items()
        }
        for m_type, subtypes in grouped_materials.items()
    }

    response = {
        'procedure': {
            'id': procedure.id,
            'procedure_id': procedure.procedure_id,
            'name': procedure.procedure_name,
            'original_cost': procedure.original_cost,
            'optimized_cost': procedure.optimized_cost
        },
        'surgeons': surgeons,
        'materials': materials_grouped
    }

    return jsonify(response)
