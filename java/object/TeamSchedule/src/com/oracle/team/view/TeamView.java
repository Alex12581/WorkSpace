package com.oracle.team.view;

import com.oracle.team.domain.Employee;
import com.oracle.team.service.NameListService;
import com.oracle.team.service.TeamService;

public class TeamView {

	private NameListService listSvc = new NameListService();
	private TeamService teamSvc = new TeamService();
	
	
	public static void main(String[] args) {
		
		TeamView util = new TeamView();
		util.enterMainMenu();
	}
	
	
	public void enterMainMenu() {
		
		listAllEmployees();

		boolean loopFlag = true;
		while(loopFlag) {
			System.out.print("1-�Ŷ��б�  2-����Ŷӳ�Ա  3-ɾ���Ŷӳ�Ա 4-�˳�   ��ѡ��(1-4)�� _");
			
			switch (TSUtility.readMenuSelection()) {
			case '1':
				System.out.println("��δ����");
				break;
			case '2':
				System.out.println("��δ����");
				break;
			case '3':
				System.out.println("��δ����");
				break;
			case '4':
				System.out.print("ȷ���Ƿ��˳�(Y/N)");
				if (TSUtility.readConfirmSelection() == 'Y') {
					loopFlag = false;
				}
				break;
			default:
				break;
			}
		}
		System.out.println("���˳�");
		
	}
	
	private void listAllEmployees() {
		
		Employee[] allEmployees = listSvc.getAllEmployees();
		
		System.out.println("-------------------------------------�����Ŷӵ������-------------------------------------- \n");
		System.out.println("ID\t����\t����\t����\tְλ\t״̬\t����\t��Ʊ\t�����豸\n");
		
		for (Employee employee : allEmployees) {
			System.out.println(employee.toString());
		}
		
		System.out.println("--------------------------------------------------------------------------------------------------- ");
		
	}
}
