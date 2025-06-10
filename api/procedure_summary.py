from flask import Blueprint, jsonify, request
from extensions import db
from models import Speciality, Procedure

procedure_summary_bp = Blueprint("procedure_summary", __name__)

@procedure_summary_bp.route("/procedure-costs-summary", methods=["GET"])
def procedure_costs_summary():
    speciality_id = request.args.get("speciality_id")

    if not speciality_id:
        return jsonify({"error": "Missing 'speciality_id' query parameter"}), 400

    try:
        speciality_id = int(speciality_id)
    except ValueError:
        return jsonify({"error": "'speciality_id' must be an integer"}), 400

    procedures = Procedure.query.filter_by(speciality_id=speciality_id).all()

    if not procedures:
        return jsonify({"message": "No procedures found for this speciality."}), 404

    result = [
        {
            "id": p.id,
            "procedure_id": p.procedure_id,
            "procedure_name": p.procedure_name,
            "original_cost": p.original_cost,
            "optimized_cost": p.optimized_cost
        }
        for p in procedures
    ]
    return jsonify(result), 200