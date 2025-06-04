import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Armchairs
{
	public static void main (String[] args) throws java.lang.Exception
	{
		Scanner sc = new Scanner(System.in);
		PrintWriter out = new PrintWriter(System.out);
		
		int n = sc.nextInt();
		Vector<Integer> a = new Vector();
		Vector<Integer> b = new Vector();
		    
		for (int i=0; i<n; ++i){
		    int input = sc.nextInt();
		    if(input==1) a.add(i);
		    else b.add(i);
		}
		
		n = a.size();
		int m = b.size();
		
		if(n==0) out.print(0);
		else{
		    int [][]dp = new int [n][m];
    		for (int i=0; i<n; ++i) for (int j=0; j<m; ++j) dp[i][j] = 25000000;
    		dp[0][0] = Math.abs(a.get(0) - b.get(0));
    		for (int i=0; i<n; ++i) for (int j=1; j<m; ++j){
    		    if(i==0) dp[i][j] = Math.min(Math.abs(a.get(i) - b.get(j)),dp[i][j-1]);
    		    else dp[i][j] = Math.min(Math.abs(a.get(i) - b.get(j)) + dp[i-1][j-1],dp[i][j-1]);
    		}
    		out.print(dp[n-1][m-1]);
		}
		
		out.close();
	}
	
	/*Methods*/
    
    
    
    
    
}
