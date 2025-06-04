import  java.math.*;
import java.util.*;
import java.lang.*;
import java.io.*;



public final class Main {

    FastReader s;


    public static void main (String[] args) throws java.lang.Exception
    {

        new Main().run();

    }
    void run()
    {
        s = new FastReader();
        solve();
    }
   StringBuffer sb;
   // int counter;
    void solve()
    {
         sb = new StringBuffer();

        for(int T = s.nextInt();T > 0;T--)
        {


            start();


        }

       // System.out.print(sb);

    }


    void start()
    {

        int n = s.nextInt();

         int mat[][] = new int[n][5];
        for(int i = 0; i<n; i++)
        {
            char [] x = s.nextLine().toCharArray();
            for(char c : x)
            {
                mat[i][c-'a']++;
            }
            int p[] = new int[5];
            int to = 0;
            for(int j : mat[i])
                to+=j;

            for(int j = 0; j<5; j++)
            {
                mat[i][j]=2*mat[i][j]-to;
            }
        }

        int ans  = 0;
        for(int i = 0; i<5; i++)
        {
            int g = check(i,mat, n);
            ans = Math.max(g,ans);
        }
        System.out.println(ans);



    }

    int check(int i, int mat[][], int n)
    {
        ArrayList<Integer> x = new ArrayList<>();
        for(int j = 0; j<n; j++)
        {
            x.add(mat[j][i]);
        }
        Collections.sort(x);
        int s = 0;
        int cnt = 0;
        int l = n-1;
        while(l>=0)
        {
            int u = x.get(l);
            if(s+u>0)
            {
                s+=u;
                cnt++;
            }
            else
                break;
            l--;
        }
        return cnt;
    }




    
    

    

    


    static void sort(int[] A){
        int n = A.length;
        Random rnd = new Random();
        for(int i=0; i<n; ++i){
            int tmp = A[i];
            int randomPos = i + rnd.nextInt(n-i);
            A[i] = A[randomPos];
            A[randomPos] = tmp;
        }
        Arrays.sort(A);
    }
    static void sort(long[] A){
        int n = A.length;
        Random rnd = new Random();
        for(int i=0; i<n; ++i){
            long tmp = A[i];
            int randomPos = i + rnd.nextInt(n-i);
            A[i] = A[randomPos];
            A[randomPos] = tmp;
        }
        Arrays.sort(A);
    }
    static String sort(String s) {
        Character ch[]=new Character[s.length()];
        for(int i=0;i<s.length();i++) {
            ch[i]=s.charAt(i);
        }
        Arrays.sort(ch);
        StringBuffer st=new StringBuffer("");
        for(int i=0;i<s.length();i++) {
            st.append(ch[i]);
        }
        return st.toString();
    }

    //long array input
    
    // int arr input
    

    



}
