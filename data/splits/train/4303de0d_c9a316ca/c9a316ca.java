import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;



import static javax.swing.UIManager.get;
import static javax.swing.UIManager.getString;


public class Main {


    static class Pair implements Comparable<Pair> {
        int x = 0;
        int y = 0;

        public Pair(int x1, int y1) {
            x = x1;
            y = y1;
        }


        
    }

    


    

    

    static int min = Integer.MAX_VALUE;

    public static void main(String[] args) throws IOException {
        FastReader sc = new FastReader();
//(1)very very important**(never take the first problem for granted, always check the test cases) take 5 minutes more and check the edge cases
        // 5 minutes will not decreases rating as much as a wrong submission does it is easy u just think with an open mind and u will surely get the answer
        //(2)let ur brain consume the problem don't just jump to the solution. after reading the problem take a pause 1 minute
        //(3)go through the example test cases and also at least two of ur own test cases.Think of testcases which are difficult(edge cases).dry run ur concept
 //(4) sometimes if else condition is not required but due to if else you miss some points and get wrong answer


        int t = sc.nextInt();
        while (t-- > 0) {

            int n =sc.nextInt();
            ArrayList<ArrayList<Integer>> list = new ArrayList<>();
            for(int i=0;i<n;i++)
                list.add(new ArrayList<Integer>());

            ArrayList<Pair> list1 = new ArrayList<>();

            for(int i=0;i<n;i++)
            list1.add(new Pair(sc.nextInt(),sc.nextInt()));

            for(int i=0;i<n-1;i++)
            {
                int a =sc.nextInt()-1,b=sc.nextInt()-1;
                list.get(a).add(b);
                list.get(b).add(a);
            }

            long[][] dp = new long[2][n];

            dfs(0,-1,dp,list,list1);


System.out.println(Math.max(dp[0][0],dp[1][0]));
        }



}

static void dfs(int u,int p,long[][] dp,ArrayList<ArrayList<Integer>> list,ArrayList<Pair> list1)
{
    for(int v:list.get(u))
    {  if(v==p)
        continue;
        dfs(v,u,dp,list,list1);
        dp[1][u]+= Math.max(Math.abs(list1.get(u).y-list1.get(v).x)+dp[0][v],Math.abs(list1.get(u).y-list1.get(v).y)+dp[1][v]);
        dp[0][u]+=Math.max(Math.abs(list1.get(u).x-list1.get(v).x)+dp[0][v],Math.abs(list1.get(u).x-list1.get(v).y)+dp[1][v]);

    }

}


//static int lcs( int[] X, int[] Y, int m, int n )
//{
//    int L[][] = new int[m+1][n+1];
//
//    /* Following steps build L[m+1][n+1] in bottom up fashion. Note
//        that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] */
//    for (int i=0; i<=m; i++)
//    {
//        for (int j=0; j<=n; j++)
//        {
//            if (i == 0 || j == 0)
//                L[i][j] = 0;
//            else if (X[i-1] == Y[j-1])
//                L[i][j] = L[i-1][j-1] + 1;
//            else
//                L[i][j] = Math.max(L[i-1][j], L[i][j-1]);
//        }
//    }
//    return L[m][n];
//}























// syntax of conditional operator y=(x==1)?1:0;
//Things to check when u r getting wrong answer
// 1-  check the flow of the code
//2- If ur stuck read the problem once again
//3- before submitting always check the output format of ur code
//4- don't check standings until problem B is done
//5- if u r thinking ur concept is correct but still u r getting wrong answer try to implement it in another way
//6- By default, java interpret all numeral literals as 32-bit integer values.
// If you want to explicitely specify that this is something bigger then 32-bit integer you should use suffix L for long values.    example   long a = 600851475143L

//All the functions




    

    










    

    //collections.sort use merge sort instead of quick sort but arrays.sort use quicksort whose worst time complexity is O(n^2)
    

    //function to print an array for debugging
    

    


    //normal gcd function, always put the greater number as a and the smaller number as b
    

    


    //to find gcd and lcm for numbers of long data type
    

    


    //Input Reader to read faster input
    



}
