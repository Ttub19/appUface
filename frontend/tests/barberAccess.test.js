import test from"node:test";import assert from"node:assert/strict";import{approvedBarber,canCreateShop,canManageShops}from"../js/barberAccess.js";
test("pending cannot manage",()=>assert.equal(canManageShops({role:"member",barber_verification_status:"pending"}),false));
test("approved barber manages",()=>{const u={role:"barber",barber_verification_status:"approved"};assert.equal(approvedBarber(u),true);assert.equal(canCreateShop(u),true)});
test("admin manages but does not create",()=>{const u={role:"admin",barber_verification_status:"none"};assert.equal(canManageShops(u),true);assert.equal(canCreateShop(u),false)});
