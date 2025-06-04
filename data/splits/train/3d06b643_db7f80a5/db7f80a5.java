import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;

public class CodeForces {

    public static void main(String[] args) throws IOException {
        FastReader input = new FastReader();
        PrintWriter output = new PrintWriter(System.out);
        //BufferedReader bf = new BufferedReader(new FileReader("input.txt"));
        //PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("output.txt")));
        //StringTokenizer stk = new StringTokenizer(bf.readLine());
        //int n=Integer.parseInt(stk.nextToken());
        int n=input.nextInt();
        ArrayList<Integer>seated=new ArrayList<>();
        ArrayList<Integer>empty=new ArrayList<>();
        for(int i=0;i<n;i++){
            int x=input.nextInt();
            if(x==1)
                seated.add(i);
            else
                empty.add(i);
        }
        if(seated.size()==0)
            output.println(0);
        else{
            output.println(helper(seated,empty));
        }
        output.close();
    }
    public static long helper(ArrayList<Integer>seated,ArrayList<Integer>empty){
        long dp[][]=new long[seated.size()+1][empty.size()+1];
        for(int i=1;i<= seated.size();i++){
            dp[i][i]=dp[i-1][i-1]+Math.abs(seated.get(i-1)-empty.get(i-1));
            for(int j=i+1;j<= empty.size();j++){
                dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(seated.get(i-1)-empty.get(j-1)));
            }
        }
        return dp[seated.size()][empty.size()];

    }







}







//Undirected Graph


//Directed non-weighted Graph





