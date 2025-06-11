from flask import Blueprint, jsonify, request
from extensions import db
from models import Procedure, Surgeon, ProcedureSurgeon

#all procedures of a surgeon
procedure_bp = Blueprint("procedure", __name__)

@procedure_bp.route("/procedures-by-surgeon", methods=["GET"])
def get_procedures_by_surgeon():
    surgeon_id = request.args.get("surgeon_id")

    if not surgeon_id:
        return jsonify({"error": "Missing 'surgeon_id' query parameter"}), 400

    try:
        surgeon_id = int(surgeon_id)
    except ValueError:
        return jsonify({"error": "'surgeon_id' must be an integer"}), 400

    # join Procedure → ProcedureSurgeon → Surgeon, filtering on your surgeon_id
    procedures = (
        Procedure.query
        .join(ProcedureSurgeon, Procedure.id == ProcedureSurgeon.procedure_id)
        .filter(ProcedureSurgeon.surgeon_id == surgeon_id)
        .all()
    )

    if not procedures:
        return jsonify({"message": "No procedures found for that surgeon."}), 404

    result = [
        {
            "id": p.id,                        # internal PK
            "procedure_id": p.procedure_id,    
            "procedure_name": p.procedure_name,
            "original_cost": p.original_cost,
            "optimized_cost": p.optimized_cost,
            "speciality_id": p.speciality_id,
            "created_at": p.created_at.isoformat()
        }
        for p in procedures
    ]

    return jsonify(result), 200
