import java.io.*;
import java.util.*;

public class Main {
	static BufferedReader br;
	static PrintWriter pr;
	static int cin() throws Exception
	{
		return Integer.valueOf(br.readLine());
	}
	static int[] split() throws Exception
	{
		String[] cmd=br.readLine().split(" ");
		int[] ans=new int[cmd.length];
		for(int i=0;i<cmd.length;i++)
		{
			ans[i]=Integer.valueOf(cmd[i]);
		}
		return ans;
	}
	
	
	
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub 
		br=new BufferedReader(new InputStreamReader(System.in));
		pr=new PrintWriter(new OutputStreamWriter(System.out));
		int cases=cin();
		while(cases!=0)
		{
			cases--;
			int[]ar=split();
		    int n=ar[0];
		    int m=ar[1];
		    int[][]mat=new int[n][m];
		    for(int i=0;i<n;i++)
		    {
		    	String s=br.readLine();
		    	for(int j=0;j<m;j++)
		    	{
		    		mat[i][j]=s.charAt(j)-'0';
		    	}
		    }
		    ArrayList<String>ans=new ArrayList<>();
		    for(int i=0;i<n-1;i++)
		    {
		    	for(int j=0;j<m-1;j++)
		    	{
		    		if(mat[i][j]==1 && mat[i][j+1]==0 && mat[i+1][j]==0 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==0 && mat[i+1][j]==0 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==1 && mat[i+1][j]==0 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==0 && mat[i+1][j]==1 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==1 && mat[i][j+1]==1 && mat[i+1][j]==1 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    		}
		    		else if(mat[i][j]==1 && mat[i][j+1]==0 && mat[i+1][j]==1 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==1 && mat[i][j+1]==1 && mat[i+1][j]==0 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==1 && mat[i+1][j]==1 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==1 && mat[i][j+1]==1 && mat[i+1][j]==1 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		
		    		else if(mat[i][j]==1 && mat[i][j+1]==1 && mat[i+1][j]==0 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+2)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==0 && mat[i+1][j]==1 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    		}
		    		else if(mat[i][j]==1 && mat[i][j+1]==0 && mat[i+1][j]==1 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==1 && mat[i+1][j]==0 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==1 && mat[i][j+1]==0 && mat[i+1][j]==0 && mat[i+1][j+1]==1)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1));
		    			ans.add((i+1)+" "+(j+2)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		else if(mat[i][j]==0 && mat[i][j+1]==1 && mat[i+1][j]==1 && mat[i+1][j+1]==0)
		    		{
		    			ans.add((i+1)+" "+(j+1)+" "+(i+1)+" "+(j+2)+" "+(i+2)+" "+(j+2));
		    			ans.add((i+1)+" "+(j+1)+" "+(i+2)+" "+(j+1)+" "+(i+2)+" "+(j+2));
		    		}
		    		mat[i][j]=mat[i+1][j]=mat[i][j+1]=mat[i+1][j+1]=0;
		    	}
		    }
		    System.out.println(ans.size());
		    for(int i=0;i<ans.size();i++)
		    {
		    	System.out.println(ans.get(i));
		    }
		}
	}
}
