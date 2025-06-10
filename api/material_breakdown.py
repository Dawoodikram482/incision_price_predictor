from flask import Blueprint, request, jsonify
from models import ProcedureMaterial, Material
from extensions import db

material_breakdown_bp = Blueprint('material_breakdown', __name__)

@material_breakdown_bp.route('/material-costs-breakdown', methods=['GET'])
def material_costs_breakdown():
    procedure_id = request.args.get('procedure_id', type=int)
    if not procedure_id:
        return jsonify({"error": "procedure_id is required"}), 400

    # Join ProcedureMaterial with Material
    procedure_materials = (
        db.session.query(Material)
        .join(ProcedureMaterial, Material.id == ProcedureMaterial.material_id)
        .filter(ProcedureMaterial.procedure_id == procedure_id)
        .all()
    )

    response = [
        {
            "material_name": material.name,
            "original_cost": material.original_price,
            "optimized_cost": material.optimized_price,
            "type": material.type,
            "subtype": material.subtype
        }
        for material in procedure_materials
    ]

    return jsonify(response)