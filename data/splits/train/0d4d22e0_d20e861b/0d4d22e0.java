import java.util.*;
public class CodeForces {
	static ArrayList<String>res;
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int test = input.nextInt();
		while(test-->0){
			int n = input.nextInt();
			int m = input.nextInt();
			char arr[][] = new char[n][m];
			res = new ArrayList<>();
			for(int i =0;i<n;i++){
				for(int j =0;j<1;j++){
					arr[i] = input.next().toCharArray();
				}
			}
			for(int i =0;i<n-1;i++){
				for(int j =0;j<m-1;j++){
					int count =0;
					if(arr[i+1][j] == '1')count++;
					if(arr[i][j+1] == '1')count++;
					if(arr[i][j] == '1')count++;
					if(arr[i+1][j+1] == '1')count++;
					if(count == 0)continue;
					if(count == 1)one(arr,i,j);
					if(count == 2)two(arr,i,j);
					if(count == 3)three(arr,i,j);
					if(count == 4)four(arr,i,j);
				}
			}
			System.out.println(res.size());
			for(String it:res)System.out.println(it);
		}	
	}
	public static void one(char arr[][],int i,int j){
		int row = i+2;
		int col = j+2;
		int count =0;
		StringBuilder sb = new StringBuilder();
		for(int x=i;x<row;x++){
			for(int y = j;y<col;y++){
				if(arr[x][y] == '1' ){
					sb.append(x+1);
					sb.append(" ");
					sb.append(y+1);
					arr[x][y] = '0';
					sb.append(" ");
				}else if(arr[x][y] == '0' && count<2){
					sb.append(x+1);
					sb.append(" ");
					sb.append(y+1);
					arr[x][y] = '1';
					sb.append(" ");
					count++;
				}
			}
		}
		res.add(sb.toString());
		two(arr,i,j);
	}
	public static void two(char arr[][],int i,int j){
			int row = i+2;
			int col = j+2;
			boolean ok = true;
			StringBuilder sb = new StringBuilder();
			for(int x=i;x<row;x++){
				for(int y = j;y<col;y++){
					if(arr[x][y] == '1' && ok){
						sb.append(x+1);
						sb.append(" ");
						sb.append(y+1);
						arr[x][y] = '0';
						sb.append(" ");
						ok = false;
					}else if(arr[x][y] == '0'){
						sb.append(x+1);
						sb.append(" ");
						sb.append(y+1);
						arr[x][y] = '1';
						sb.append(" ");
					}
				}
			}
			res.add(sb.toString());
			three(arr,i,j);
		}
	public static void three(char arr[][],int i,int j){
		StringBuilder sb = new StringBuilder();
		if(arr[i][j] == '1'){
			sb.append(i+1);
			sb.append(" ");
			sb.append(j+1);
			arr[i][j] = '0';
			sb.append(" ");
		}
		if(arr[i+1][j] == '1'){
			sb.append(i+2);
			sb.append(" ");
			sb.append(j+1);
			arr[i+1][j] = '0';
			sb.append(" ");
		}
		if(arr[i][j+1] == '1'){
			arr[i][j+1] ='0';
			sb.append(i+1);
			sb.append(" ");
			sb.append(j+2);
			sb.append(" ");
		}
		if(arr[i+1][j+1] == '1'){
			arr[i+1][j+1] = '0';
			sb.append(i+2);
			sb.append(" ");
			sb.append(j+2);
		}
		res.add(sb.toString());
	}
	public static void four(char arr[][],int i,int j){
		int row = i+2;
		int col = j+2;
		int count =0;
		StringBuilder sb = new StringBuilder();
		for(int x=i;x<row;x++){
			for(int y = j;y<col;y++){
				if(count<3){
					sb.append(x+1);
					sb.append(" ");
					sb.append(y+1);
					arr[x][y] = '0';
					sb.append(" ");
					count++;
				}
			}
		}
		res.add(sb.toString());
		one(arr,i,j);
	}
}
