import java.util.*;
import java.io.*;

public class code{

	

	

	public static void main(String[] arg) throws IOException{
		//Reader in=new Reader();
		Scanner in=new Scanner(System.in);

   int n=in.nextInt();
	 int[] arr=new int[n];
	 ArrayList<Integer> zero=new ArrayList<Integer>();
	 ArrayList<Integer> one=new ArrayList<Integer>();

	 for(int i=0;i<n;i++){
		 arr[i]=in.nextInt();
		 if(arr[i]==0) zero.add(i);
		 else one.add(i);
	 }
   if(one.size()==0) {
		 System.out.println(0);
	 }
	 else{
			 int[][] dp=new int[one.size()][zero.size()];

		   for(int i=0;i<one.size();i++){
				 for(int j=0;j<zero.size();j++){
					 if(i==0 && j==0) dp[i][j]=Math.abs(one.get(i)-zero.get(j));
					 else if(j==0) dp[i][j]=Integer.MAX_VALUE/2;
					 else if(i==0) dp[i][j]=Math.min(dp[i][j-1],Math.abs(one.get(i)-zero.get(j)));
					 else{
						 dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(one.get(i)-zero.get(j)));
					 }
				 }
			 }

		  System.out.println(dp[one.size()-1][zero.size()-1]);
		}
	}
}
