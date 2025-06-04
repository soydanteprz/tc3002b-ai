import java.util.*;
import java.io.*;
public class Armchairs{
    public static void main(String[] args) {
        FastReader fr = new FastReader();
        PrintWriter out = new PrintWriter(System.out);
        Scanner sc= new Scanner (System.in);
        
        //Code From Here----
        int t =fr.nextInt();
        ArrayList <Integer> chairs= new ArrayList<>();
        ArrayList <Integer> free= new ArrayList<>();

        for (int i = 0; i < t; i++) {
            int state =fr.nextInt();
            if(state==1)
            {
                chairs.add(i);
            }
            else{
                free.add(i);
            }
        }
        int [][] dp=new int [t][t];
        for (int[] is : dp) {
            Arrays.fill(is, -1);
        }
        int ans=solve(chairs,free,0,0,chairs.size(),dp);
        out.println(ans);
        
    
        out.flush();
        sc.close();
    }

    //This RadixSort() is for long method

   

    private static int solve(ArrayList<Integer> chairs, ArrayList<Integer> free, int i, int j, int size,int [][] dp) {
        if (dp[i][j]!=-1) {
            return dp[i][j];
        }
        if (size==0) {
            return 0;
        }
        if (j==free.size()) {
            return 10000000;
        }
        int a=solve(chairs, free, i, j+1, size,dp);
        int b=Math.abs(chairs.get(i)-free.get(j))+solve(chairs, free, i+1, j+1, size-1,dp);
        dp[i][j]=Math.min(a, b);
        return dp[i][j];
    }

    
	

	

	


	
    

    

    

    


	// For Fast Input  ----
    
    
	



    
}
